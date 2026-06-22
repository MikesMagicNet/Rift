#!/usr/bin/env python3
"""
Rift — Tool & Context Engineering Layer

The Tool Execution Layer is built around a ToolRegistry that dispatches calls
to typed handlers covering file operations, process execution, and web access, with support for batch
parallel execution and on-demand MCP tool discovery (Section 2.4.7). A Skills system lazily injects reusable,
domain-specific prompt templates from a three-tier hierarchy (built-in, project, user). The Context Engineering
Layer manages the LLM context window through four subsystems: System Reminders (Section 2.3.4) for
context-aware behavioral guidance, Prompt Composer for modular system-prompt assembly, Memory for
cross-session continuity, and Compaction (Section 2.3.6) for reclaiming token budget.

This module builds ON TOP of agentic_layer.py. The dependency flows one way:
    cli.py → agentic_layer.py → tools_plus_context.py
agentic_layer owns the core primitives (ToolRegistry, ContextCompactor, AgentMode,
ModelClientPool, the built-in tool handlers). This module adds the remaining
Tool & Context subsystems and the meta-tools (invoke_skill, search_tools,
batch_tool) that plug into the existing registry.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

# Shared primitives from the agent core layer.
from harness.agentic_layer import (
    AgentMode,
    ContextCompactor,
    ToolRegistry,
)

if TYPE_CHECKING:
    from cli import ApprovalManager, ConfigManager, ModeManager, SessionManager


PROJECT_ROOT = Path(__file__).resolve().parent

# Batch execution: paper specifies max 5 concurrent workers (Section 2.4.8)
MAX_BATCH_WORKERS = 5


# ═════════════════════════════════════════════════════════════════════
#  ToolExecutionContext — cross-cutting services bundle (Section 2.4.1)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ToolExecutionContext:
    """Bundles cross-cutting services injected into every tool handler.

    Mirrors the paper's ToolExecutionContext (Section 2.4.1): mode manager,
    approval manager, session manager, plus Rift's context subsystems so
    meta-tools (invoke_skill, search_tools, batch_tool) can reach back into
    the surrounding machinery.
    """
    mode_manager: Any = None
    approval_manager: Any = None
    session_manager: Any = None
    config_manager: Any = None
    tool_registry: Optional[ToolRegistry] = None
    skill_loader: Optional["SkillLoader"] = None
    mcp_registry: Optional["MCPRegistry"] = None
    memory_store: Optional["MemoryStore"] = None
    ui_callback: Optional[Callable[[str], None]] = None
    # Tracks last-read mtimes for stale-read detection (Section 2.4.2)
    file_time_tracker: dict[str, float] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════
#  PromptComposer — modular system-prompt assembly (Section 2.3.1)
# ═════════════════════════════════════════════════════════════════════

class PromptTier(Enum):
    """Five functional tiers; lower value = earlier in the prompt."""
    CORE_IDENTITY = 10
    TOOL_DEFINITIONS = 20
    SAFETY_RULES = 30
    PROVIDER_GUIDANCE = 40
    DYNAMIC_CONTEXT = 50


@dataclass
class PromptSection:
    """One modular instruction block.

    condition: predicate over the runtime context dict; None = always include.
    priority:  ascending sort key (lower appears earlier).
    """
    name: str
    content: str
    tier: PromptTier = PromptTier.SAFETY_RULES
    priority: int = 100
    condition: Optional[Callable[[dict[str, Any]], bool]] = None


class PromptComposer:
    """Priority-ordered conditional composition of the system prompt.

    Four-step pipeline (Section 2.3.1):
      1. Filter   — drop sections whose predicate returns False
      2. Sort     — ascending (tier, priority)
      3. Load     — resolve ${VAR} placeholders from the name registry
      4. Join     — concatenate into the final prompt
    """

    _VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

    def __init__(self, core_role: str = "") -> None:
        self.core_role = core_role
        self._sections: list[PromptSection] = []
        self._var_registry: dict[str, str] = {}

    def register(self, section: PromptSection) -> None:
        self._sections.append(section)

    def register_var(self, name: str, value: str) -> None:
        """Register a ${VAR} substitution (decouples prose from tool names)."""
        self._var_registry[name] = value

    def _resolve_vars(self, text: str) -> str:
        def repl(m: "re.Match[str]") -> str:
            return self._var_registry.get(m.group(1), m.group(0))
        return self._VAR_RE.sub(repl, text)

    def compose(self, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}

        # 1. Filter
        surviving = [
            s for s in self._sections
            if s.condition is None or s.condition(ctx)
        ]

        # 2. Sort by (tier value, priority)
        surviving.sort(key=lambda s: (s.tier.value, s.priority))

        # 3. Load + resolve vars
        rendered = [self._resolve_vars(s.content) for s in surviving]

        # 4. Join: core role first, then sections, then dynamic env block
        parts: list[str] = []
        if self.core_role:
            parts.append(self._resolve_vars(self.core_role))
        parts.extend(rendered)

        env_block = self._build_env_block(ctx)
        if env_block:
            parts.append(env_block)

        return "\n\n".join(p for p in parts if p.strip())

    def _build_env_block(self, ctx: dict[str, Any]) -> str:
        """Dynamically collected environment metadata (Dynamic Context tier)."""
        lines = []
        if ctx.get("cwd"):
            lines.append(f"Working directory: {ctx['cwd']}")
        if ctx.get("in_git_repo"):
            lines.append("Git repository: yes")
        if ctx.get("skills_index"):
            lines.append(f"Available skills:\n{ctx['skills_index']}")
        if not lines:
            return ""
        return "## Environment\n" + "\n".join(lines)

    @classmethod
    def default_action_composer(cls, core_role: str = "") -> "PromptComposer":
        """Build a composer pre-loaded with Rift's default action-mode sections."""
        composer = cls(core_role or DEFAULT_CORE_ROLE)

        composer.register(PromptSection(
            name="identity",
            content=(
                "You are Rift, an autonomous terminal-native coding agent. You reason "
                "carefully, prefer safe actions, and verify your work before declaring "
                "a task complete."
            ),
            tier=PromptTier.CORE_IDENTITY,
            priority=10,
        ))
        composer.register(PromptSection(
            name="tool_use",
            content=(
                "Use ${READ_TOOL} before editing any file. Prefer ${SEARCH_TOOL} over "
                "broad directory listing. Batch independent reads with ${BATCH_TOOL}."
            ),
            tier=PromptTier.TOOL_DEFINITIONS,
            priority=20,
        ))
        composer.register(PromptSection(
            name="git_workflow",
            content=(
                "This directory is a git repository. Make atomic commits with clear "
                "messages. Never force-push. Confirm before destructive git operations."
            ),
            tier=PromptTier.SAFETY_RULES,
            priority=30,
            condition=lambda c: bool(c.get("in_git_repo")),
        ))
        composer.register(PromptSection(
            name="task_tracking",
            content=(
                "Track multi-step work as a todo list. Do not declare completion while "
                "items remain open."
            ),
            tier=PromptTier.SAFETY_RULES,
            priority=35,
            condition=lambda c: bool(c.get("task_tracking_enabled")),
        ))
        composer.register(PromptSection(
            name="error_recovery",
            content=(
                "On tool failure: classify the error, re-read relevant state, and retry "
                "with a corrected approach before giving up."
            ),
            tier=PromptTier.SAFETY_RULES,
            priority=40,
        ))

        # Default var registry: decouple prose from concrete tool names.
        composer.register_var("READ_TOOL", "read_file")
        composer.register_var("SEARCH_TOOL", "search_files")
        composer.register_var("BATCH_TOOL", "batch_tool")
        return composer


DEFAULT_CORE_ROLE = (
    "# Rift Agent\nYou operate autonomously in a terminal environment with access "
    "to file, shell, and web tools. Favor correctness and safety over speed."
)


# ═════════════════════════════════════════════════════════════════════
#  System Reminders — event-driven behavioral guidance (Section 2.3.4)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ReminderTemplate:
    """A single named reminder with a firing cap."""
    name: str
    template: str
    max_fires: int = 3  # guardrail counter cap


class SystemReminderManager:
    """Event-driven reminder injection layer (Section 2.3.4).

    After each ReAct iteration, eight event detectors examine conversation
    state. When one fires, its template is resolved, checked against a
    guardrail counter, and returned as a role:user message placed at maximum
    recency (immediately before the next LLM call).
    """

    def __init__(self) -> None:
        self._templates: dict[str, ReminderTemplate] = {}
        self._fire_counts: dict[str, int] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            ReminderTemplate(
                "tool_failure_no_retry",
                "[REMINDER] The last tool call failed and you have not retried. "
                "Classify the error and attempt a corrected approach.",
            ),
            ReminderTemplate(
                "exploration_spiral",
                "[REMINDER] You have performed {count} consecutive reads without acting. "
                "Consolidate what you've learned and take a concrete step.",
            ),
            ReminderTemplate(
                "denied_reattempt",
                "[REMINDER] A previously denied tool call is being retried unchanged. "
                "Do not repeat it; choose a different approach.",
            ),
            ReminderTemplate(
                "premature_completion",
                "[REMINDER] You signaled completion but these todos remain open:\n{todo_list}\n"
                "Finish them before stopping.",
            ),
            ReminderTemplate(
                "work_after_done",
                "[REMINDER] All todos are complete. Stop and summarize rather than "
                "continuing to act.",
            ),
            ReminderTemplate(
                "plan_no_followthrough",
                "[REMINDER] A plan was approved but not yet executed. Begin implementing it.",
            ),
            ReminderTemplate(
                "unprocessed_subagent",
                "[REMINDER] Subagent results are available but unprocessed. Synthesize "
                "them into your response.",
            ),
            ReminderTemplate(
                "empty_completion",
                "[REMINDER] Your completion message was empty. Provide a substantive "
                "summary of what you accomplished.",
            ),
        ]
        for t in defaults:
            self._templates[t.name] = t

    def _can_fire(self, name: str) -> bool:
        tmpl = self._templates.get(name)
        if tmpl is None:
            return False
        return self._fire_counts.get(name, 0) < tmpl.max_fires

    def get_reminder(self, name: str, **kwargs: Any) -> str | None:
        """Resolve a reminder template if its guardrail allows firing."""
        if not self._can_fire(name):
            return None
        tmpl = self._templates[name]
        self._fire_counts[name] = self._fire_counts.get(name, 0) + 1
        try:
            return tmpl.template.format(**kwargs)
        except KeyError:
            return tmpl.template

    def detect(self, state: "ConversationState") -> list[str]:
        """Run the eight event detectors; return reminder messages to inject."""
        out: list[str] = []

        if state.last_tool_failed and not state.retried_after_failure:
            r = self.get_reminder("tool_failure_no_retry")
            if r:
                out.append(r)

        if state.consecutive_reads >= 5:
            r = self.get_reminder("exploration_spiral", count=state.consecutive_reads)
            if r:
                out.append(r)

        if state.denied_reattempt:
            r = self.get_reminder("denied_reattempt")
            if r:
                out.append(r)

        if state.signaled_completion and state.open_todos:
            r = self.get_reminder(
                "premature_completion",
                todo_list="\n".join(f"  - {t}" for t in state.open_todos),
            )
            if r:
                out.append(r)

        if state.all_todos_done and state.still_acting:
            r = self.get_reminder("work_after_done")
            if r:
                out.append(r)

        if state.plan_approved and not state.plan_started:
            r = self.get_reminder("plan_no_followthrough")
            if r:
                out.append(r)

        if state.unprocessed_subagent_results:
            r = self.get_reminder("unprocessed_subagent")
            if r:
                out.append(r)

        if state.signaled_completion and state.empty_completion_message:
            r = self.get_reminder("empty_completion")
            if r:
                out.append(r)

        return out

    def reset(self) -> None:
        self._fire_counts.clear()


@dataclass
class ConversationState:
    """Snapshot of conversation signals the reminder detectors examine."""
    last_tool_failed: bool = False
    retried_after_failure: bool = False
    consecutive_reads: int = 0
    denied_reattempt: bool = False
    signaled_completion: bool = False
    open_todos: list[str] = field(default_factory=list)
    all_todos_done: bool = False
    still_acting: bool = False
    plan_approved: bool = False
    plan_started: bool = False
    unprocessed_subagent_results: bool = False
    empty_completion_message: bool = False


# ═════════════════════════════════════════════════════════════════════
#  MemoryStore — cross-session continuity (experience-driven, Section 2.3.6)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class MemoryBullet:
    """One playbook entry accumulated across sessions."""
    text: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class MemoryStore:
    """Experience-driven memory pipeline (ACE-style playbook).

    A Reflector analyzes outcomes and a Curator writes durable bullets to a
    JSON-backed playbook. Relevant bullets are surfaced into the action prompt.
    """

    def __init__(self, path: Path | None = None, max_bullets: int = 200) -> None:
        self.path = path or PROJECT_ROOT / ".memory.json"
        self.max_bullets = max_bullets
        self._bullets: list[MemoryBullet] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self._bullets = [
                    MemoryBullet(
                        text=b["text"],
                        tags=b.get("tags", []),
                        created_at=b.get("created_at", time.time()),
                        hits=b.get("hits", 0),
                    )
                    for b in raw.get("bullets", [])
                ]
            except (json.JSONDecodeError, KeyError):
                self._bullets = []

    def _save(self) -> None:
        out = {
            "bullets": [
                {"text": b.text, "tags": b.tags, "created_at": b.created_at, "hits": b.hits}
                for b in self._bullets
            ]
        }
        self.path.write_text(json.dumps(out, indent=2))

    def curate(self, text: str, tags: list[str] | None = None) -> None:
        """Curator: add a new lesson, dedup by text, enforce capacity."""
        text = text.strip()
        if not text:
            return
        for b in self._bullets:
            if b.text == text:
                return  # already known
        self._bullets.append(MemoryBullet(text=text, tags=tags or []))
        # Evict least-useful (lowest hits, oldest) if over capacity.
        if len(self._bullets) > self.max_bullets:
            self._bullets.sort(key=lambda b: (b.hits, b.created_at))
            self._bullets = self._bullets[-self.max_bullets:]
        self._save()

    def reflect(self, user_message: str, outcome: str, success: bool) -> None:
        """Reflector: derive a lesson from an outcome and curate it.

        Lightweight heuristic reflection (no extra LLM call): records concise,
        reusable signal rather than transcripts.
        """
        if success:
            return  # only failures/edge-cases are worth durable storage by default
        snippet = outcome.strip().splitlines()[0][:160] if outcome.strip() else ""
        if snippet:
            self.curate(f"When '{user_message[:60]}...': watch for — {snippet}", tags=["lesson"])

    def relevant_bullets(self, query: str, limit: int = 5) -> list[str]:
        """Surface the most relevant bullets for the current query."""
        if not self._bullets:
            return []
        q_tokens = {t for t in re.findall(r"[a-z0-9]{3,}", query.lower())}
        scored: list[tuple[int, MemoryBullet]] = []
        for b in self._bullets:
            b_tokens = set(re.findall(r"[a-z0-9]{3,}", b.text.lower()))
            score = len(q_tokens & b_tokens)
            if score:
                scored.append((score, b))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[:limit]
        for _score, b in chosen:
            b.hits += 1
        if chosen:
            self._save()
        return [b.text for _s, b in chosen]


# ═════════════════════════════════════════════════════════════════════
#  Skills — three-tier lazy prompt-template injection (Section 2.4.8)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class SkillMeta:
    """Lightweight skill index entry (frontmatter only)."""
    name: str
    description: str
    path: Path
    tier: str  # "project" | "user" | "builtin"


class SkillLoader:
    """Two-phase skills system (Section 2.4.8).

    Phase 1 (startup): scan three tiers, parse YAML frontmatter only, build a
                       lightweight index for the system prompt.
    Phase 2 (on-demand): invoke_skill reads the full body, strips frontmatter,
                         injects it into context. Dedup cache: load once/session.

    Priority: project (.rift/skills) > user (~/.rift/skills) > builtin.
    """

    _FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or PROJECT_ROOT
        # Lowest priority first; later registrations override earlier on name clash.
        self._tier_dirs: list[tuple[str, Path]] = [
            ("builtin", root / "skills"),
            ("user", Path.home() / ".rift" / "skills"),
            ("project", root / ".rift" / "skills"),
        ]
        self._index: "OrderedDict[str, SkillMeta]" = OrderedDict()
        self._loaded_cache: set[str] = set()
        self.discover()

    def _parse_frontmatter(self, text: str) -> dict[str, str]:
        m = self._FRONTMATTER_RE.match(text)
        if not m:
            return {}
        fields: dict[str, str] = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                fields[k.strip()] = v.strip().strip('"').strip("'")
        return fields

    def discover(self) -> None:
        """Phase 1: build the metadata index across all three tiers."""
        self._index.clear()
        for tier, d in self._tier_dirs:  # builtin → user → project (override order)
            if not d.exists():
                continue
            for skill_file in sorted(d.glob("*.md")):
                try:
                    text = skill_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                fm = self._parse_frontmatter(text)
                name = fm.get("name", skill_file.stem)
                desc = fm.get("description", "")
                # Higher-priority tier overrides same-named entry.
                self._index[name] = SkillMeta(name=name, description=desc, path=skill_file, tier=tier)

    def index_text(self) -> str:
        """Render the skill index for inclusion in the system prompt."""
        if not self._index:
            return ""
        lines = [f"  - {m.name}: {m.description}" for m in self._index.values()]
        return "\n".join(lines)

    def invoke(self, name: str) -> str:
        """Phase 2: load a skill body on demand (dedup per session)."""
        meta = self._index.get(name)
        if meta is None:
            return f"[Skill '{name}' not found. Available: {', '.join(self._index) or 'none'}]"
        if name in self._loaded_cache:
            return f"[Skill '{name}' already loaded this session.]"
        try:
            text = meta.path.read_text(encoding="utf-8")
        except OSError as e:
            return f"[Error loading skill '{name}': {e}]"
        body = self._FRONTMATTER_RE.sub("", text).strip()
        self._loaded_cache.add(name)
        return body

    def reset(self) -> None:
        self._loaded_cache.clear()


# ═════════════════════════════════════════════════════════════════════
#  MCP — token-efficient external tool discovery (Section 2.4.7)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class MCPTool:
    """An external MCP tool: schema cost only paid once discovered."""
    qualified_name: str          # e.g. mcp__github__create_issue
    description: str
    schema: dict[str, Any]
    server: str
    invoke: Optional[Callable[[dict[str, Any]], Any]] = None


class MCPRegistry:
    """Lazy MCP tool discovery via keyword search (Section 2.4.7).

    Initial context contains zero external tool schemas. The agent calls
    search_tools(query, detail) to surface relevant tools; matched tools are
    marked discovered and their schemas join subsequent LLM calls. Direct
    invocation by qualified name auto-discovers.
    """

    def __init__(self) -> None:
        self._registered: dict[str, MCPTool] = {}
        self._discovered: set[str] = set()

    def register_server_tool(self, tool: MCPTool) -> None:
        self._registered[tool.qualified_name] = tool

    def _keywords(self, text: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())}

    def search(self, query: str, detail: str = "brief", limit: int = 5) -> list[dict[str, Any]]:
        """Keyword-scored discovery. detail ∈ {names, brief, full}."""
        q_tokens = self._keywords(query)
        scored: list[tuple[int, MCPTool]] = []
        for tool in self._registered.values():
            name_tokens = self._keywords(tool.qualified_name)
            desc_tokens = self._keywords(tool.description)
            score = 2 * len(q_tokens & name_tokens) + len(q_tokens & desc_tokens)
            if score:
                scored.append((score, tool))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for _score, tool in scored[:limit]:
            if detail == "full":
                self._discovered.add(tool.qualified_name)
                results.append({
                    "name": tool.qualified_name,
                    "description": tool.description,
                    "schema": tool.schema,
                })
            elif detail == "brief":
                results.append({"name": tool.qualified_name, "description": tool.description})
            else:  # names
                results.append({"name": tool.qualified_name})
        return results

    def discover(self, qualified_name: str) -> bool:
        if qualified_name in self._registered:
            self._discovered.add(qualified_name)
            return True
        return False

    def discovered_schemas(self) -> list[dict[str, Any]]:
        """OpenAI schemas for discovered tools only (near-zero baseline cost)."""
        out = []
        for name in self._discovered:
            tool = self._registered.get(name)
            if tool:
                out.append({
                    "type": "function",
                    "function": {
                        "name": tool.qualified_name,
                        "description": tool.description,
                        "parameters": tool.schema,
                    },
                })
        return out

    def invoke(self, qualified_name: str, arguments: dict[str, Any]) -> Any:
        # Direct invocation auto-discovers (Section 2.4.7).
        tool = self._registered.get(qualified_name)
        if tool is None:
            return f"[MCP tool '{qualified_name}' not found]"
        self._discovered.add(qualified_name)
        if tool.invoke is None:
            return f"[MCP tool '{qualified_name}' has no bound handler]"
        try:
            return tool.invoke(arguments)
        except Exception as e:
            return f"[MCP invocation error: {e}]"


# ═════════════════════════════════════════════════════════════════════
#  BatchExecutor — parallel / serial multi-tool execution (Section 2.4.8)
# ═════════════════════════════════════════════════════════════════════

class BatchExecutor:
    """Run multiple tool calls in one turn (Section 2.4.8).

    parallel: thread pool, max 5 workers — independent ops (reads, searches).
    serial:   ordered — dependent ops (mkdir then write).
    The caller specifies the mode because only it knows the dependencies.
    """

    def __init__(self, tool_registry: ToolRegistry, max_workers: int = MAX_BATCH_WORKERS) -> None:
        self.tool_registry = tool_registry
        self.max_workers = max_workers

    def run(self, calls: list[dict[str, Any]], mode: str = "parallel") -> list[dict[str, Any]]:
        if mode == "serial":
            return self._run_serial(calls)
        return self._run_parallel(calls)

    def _exec_one(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name", "")
        args = call.get("arguments", {})
        try:
            result = self.tool_registry.execute(name, args)
        except Exception as e:
            result = f"[Error: {e}]"
        return {"tool": name, "result": result}

    def _run_serial(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._exec_one(c) for c in calls]

    def _run_parallel(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = [None] * len(calls)  # type: ignore[list-item]
        workers = min(self.max_workers, max(1, len(calls)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(self._exec_one, c): i for i, c in enumerate(calls)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
        return results


# ═════════════════════════════════════════════════════════════════════
#  Meta-tools — register into the existing ToolRegistry
# ═════════════════════════════════════════════════════════════════════

class InvokeSkillTool:
    name = "invoke_skill"
    description = "Load a named skill's instructions into context on demand."
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The skill name to invoke"},
        },
        "required": ["name"],
    }
    read_only = True

    def __init__(self, skill_loader: SkillLoader) -> None:
        self.skill_loader = skill_loader

    def execute(self, arguments: dict[str, Any]) -> Any:
        return self.skill_loader.invoke(arguments["name"])


class SearchToolsTool:
    name = "search_tools"
    description = "Discover external MCP tools by keyword. detail: names|brief|full."
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "detail": {"type": "string", "enum": ["names", "brief", "full"], "description": "Detail level"},
        },
        "required": ["query"],
    }
    read_only = True

    def __init__(self, mcp_registry: MCPRegistry) -> None:
        self.mcp_registry = mcp_registry

    def execute(self, arguments: dict[str, Any]) -> Any:
        results = self.mcp_registry.search(
            arguments["query"],
            detail=arguments.get("detail", "brief"),
        )
        return json.dumps(results, indent=2) if results else "[No matching tools]"


class BatchTool:
    name = "batch_tool"
    description = "Execute multiple tool calls in one turn. mode: parallel|serial."
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["parallel", "serial"]},
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name"],
                },
            },
        },
        "required": ["calls"],
    }
    read_only = False

    def __init__(self, batch_executor: BatchExecutor) -> None:
        self.batch_executor = batch_executor

    def execute(self, arguments: dict[str, Any]) -> Any:
        calls = arguments.get("calls", [])
        mode = arguments.get("mode", "parallel")
        results = self.batch_executor.run(calls, mode=mode)
        return json.dumps(results, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════
#  ContextEngine — wires the four context subsystems together (Section 2.3)
# ═════════════════════════════════════════════════════════════════════

class ContextEngine:
    """Single entry point that bundles the Context Engineering Layer's four
    subsystems (Section 2.1): System Reminders, Prompt Composer, Memory, and
    Compaction. AgenticAgent constructs one of these and the ReActExecutor
    drives it each turn.
    """

    def __init__(
        self,
        composer: PromptComposer,
        reminders: SystemReminderManager,
        memory: MemoryStore,
        compactor: ContextCompactor,
    ) -> None:
        self.composer = composer
        self.reminders = reminders
        self.memory = memory
        self.compactor = compactor

    def build_system_prompt(self, runtime_ctx: dict[str, Any]) -> str:
        return self.composer.compose(runtime_ctx)

    def memory_preamble(self, user_message: str, limit: int = 5) -> str:
        bullets = self.memory.relevant_bullets(user_message, limit=limit)
        if not bullets:
            return ""
        return "## Relevant memory\n" + "\n".join(f"  - {b}" for b in bullets)

    def reminders_for(self, state: ConversationState) -> list[str]:
        return self.reminders.detect(state)

    def reflect(self, user_message: str, outcome: str, success: bool) -> None:
        self.memory.reflect(user_message, outcome, success)


# ═════════════════════════════════════════════════════════════════════
#  Integration helper — attach context+tool subsystems to an AgenticAgent
# ═════════════════════════════════════════════════════════════════════

def build_context_engine(config: Any, compactor: ContextCompactor) -> ContextEngine:
    """Construct a fully-wired ContextEngine from a ConfigManager."""
    cwd = os.getcwd()
    in_git = (Path(cwd) / ".git").exists()

    composer = PromptComposer.default_action_composer(
        core_role=config.system_prompt or DEFAULT_CORE_ROLE
    )
    reminders = SystemReminderManager()
    memory = MemoryStore()
    engine = ContextEngine(composer, reminders, memory, compactor)
    return engine


def install_tool_and_context_layer(agent: Any) -> ToolExecutionContext:
    """Wire skills, MCP, batch execution, and meta-tools into an AgenticAgent.

    Called from AgenticAgent.__init__ after the base tool registry exists.
    Returns the ToolExecutionContext bundle for downstream use.
    """
    skill_loader = SkillLoader()
    mcp_registry = MCPRegistry()
    batch_executor = BatchExecutor(agent.tool_registry)
    memory_store = MemoryStore()

    # Register meta-tools into the existing registry.
    agent.tool_registry.register(InvokeSkillTool(skill_loader))
    agent.tool_registry.register(SearchToolsTool(mcp_registry))
    agent.tool_registry.register(BatchTool(batch_executor))

    ctx = ToolExecutionContext(
        mode_manager=getattr(agent, "mode", None),
        approval_manager=getattr(agent, "approval", None),
        session_manager=getattr(agent, "session", None),
        config_manager=getattr(agent, "config", None),
        tool_registry=agent.tool_registry,
        skill_loader=skill_loader,
        mcp_registry=mcp_registry,
        memory_store=memory_store,
    )
    return ctx
