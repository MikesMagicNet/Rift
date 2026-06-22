#!/usr/bin/env python3
"""
Rift — Agentic Layer

Assigns five specialized model roles to distinct LLMs (Section 2.2.5), each lazily
initialized and informed by a locally cached capability registry. The system operates in two modes: Normal
Mode with full read-write tool access for execution, and Plan Mode restricted to read-only tools for safe
planning (Section 2.2). Reasoning proceeds through the Extended ReAct Loop (Section 2.2.6), which runs
four phases per turn: automatic context compaction when the token budget nears exhaustion; an optional
thinking phase for pre-action reasoning at configurable depth; an optional self-critique phase; and the
standard Reason-Act-Execute-Observe action phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    from cli import ApprovalManager, ConfigManager, ModeManager, SessionManager


# ═════════════════════════════════════════════════════════════════════
#  Constants & Enums
# ═════════════════════════════════════════════════════════════════════

class ThinkingLevel(Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentMode(Enum):
    NORMAL = "normal"   # Full read-write tool access
    PLAN = "plan"         # Read-only tools only


# Compaction thresholds (Section 2.2.6 Algorithm 1)
COMPACTION_WARN = 0.70 
COMPACTION_MASK = 0.80
COMPACTION_PRUNE = 0.85
COMPACTION_AGGRESSIVE_MASK = 0.90
COMPACTION_FULL = 0.99

# Doom-loop detection
DOOM_WINDOW_SIZE = 20
DOOM_REPEAT_THRESHOLD = 3

# Safety limits
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_MAX_TOKENS = 128_000
NUDGE_BUDGET = 3


# ═════════════════════════════════════════════════════════════════════
#  Capability Registry — locally cached model metadata
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ModelCapability:
    """Cached capability metadata for a single model."""
    model_id: str
    context_length: int
    vision: bool
    supports_thinking: bool
    supports_tool_use: bool
    reasoning_effort: str | None = None
    # Epoch seconds when this entry was written
    cached_at: float = field(default_factory=time.time)

    def is_stale(self, ttl_seconds: float = 86400) -> bool:
        return time.time() - self.cached_at > ttl_seconds


class CapabilityRegistry:
    """Locally cached registry of model capabilities with stale-while-revalidate."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path or Path(__file__).resolve().parent / ".model_cache.json"
        self._cache: dict[str, ModelCapability] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                raw = json.loads(self.cache_path.read_text())
                for model_id, data in raw.items():
                    self._cache[model_id] = ModelCapability(
                        model_id=data.get("model_id", model_id),
                        context_length=data.get("context_length", DEFAULT_MAX_TOKENS),
                        vision=data.get("vision", False),
                        supports_thinking=data.get("supports_thinking", True),
                        supports_tool_use=data.get("supports_tool_use", True),
                        reasoning_effort=data.get("reasoning_effort"),
                        cached_at=data.get("cached_at", 0),
                    )
            except (json.JSONDecodeError, KeyError):
                self._cache = {}

    def _save(self) -> None:
        out = {}
        for k, cap in self._cache.items():
            out[k] = {
                "model_id": cap.model_id,
                "context_length": cap.context_length,
                "vision": cap.vision,
                "supports_thinking": cap.supports_thinking,
                "supports_tool_use": cap.supports_tool_use,
                "reasoning_effort": cap.reasoning_effort,
                "cached_at": cap.cached_at,
            }
        self.cache_path.write_text(json.dumps(out, indent=2))

    def get(self, model_id: str) -> ModelCapability | None:
        return self._cache.get(model_id)

    def get_context_length(self, model_id: str) -> int:
        cap = self._cache.get(model_id)
        return cap.context_length if cap else DEFAULT_MAX_TOKENS

    def is_vision_capable(self, model_id: str) -> bool:
        cap = self._cache.get(model_id)
        return cap.vision if cap else False

    def put(self, cap: ModelCapability) -> None:
        self._cache[cap.model_id] = cap
        self._save()

    def bootstrap_defaults(self) -> None:
        """Seed the registry with known NVIDIA NIM defaults."""
        defaults = {
            "nvidia/nemotron-3-ultra-550b-a55b": ModelCapability(
                model_id="nvidia/nemotron-3-ultra-550b-a55b",
                context_length=128_000,
                vision=False,
                supports_thinking=True,
                supports_tool_use=True,
            ),
            "nvidia/nemotron-3-super-120b-a12b": ModelCapability(
                model_id="nvidia/nemotron-3-super-120b-a12b",
                context_length=128_000,
                vision=False,
                supports_thinking=True,
                supports_tool_use=True,
            ),
            "nvidia/nemotron-4-340b-instruct": ModelCapability(
                model_id="nvidia/nemotron-4-340b-instruct",
                context_length=128_000,
                vision=False,
                supports_thinking=True,
                supports_tool_use=True,
            ),
            "meta/llama-3.2-90b-vision-instruct": ModelCapability(
                model_id="meta/llama-3.2-90b-vision-instruct",
                context_length=128_000,
                vision=True,
                supports_thinking=True,
                supports_tool_use=True,
            ),
        }
        for model_id, cap in defaults.items():
            if model_id not in self._cache:
                self._cache[model_id] = cap
        self._save()


# ═════════════════════════════════════════════════════════════════════
#  Five-Slot Client Pool — lazy init per model role (Section 2.2.5)
# ═════════════════════════════════════════════════════════════════════

class LazyLLMClient:
    """Wraps an OpenAI-compatible client with lazy initialization."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def _init(self) -> Any:
        from openai import OpenAI
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._init()
        return self._client

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096, tools: list[dict] | None = None, tool_choice: str | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)


class ModelRole(Enum):
    ACTION = "action"
    THINKING = "thinking"
    CRITIQUE = "critique"
    VISION = "vision"
    COMPACT = "compact"


class ModelClientPool:
    """Holds five lazy-initialized clients with fallback chains (Section 2.2.5).

    Fallback chains:
      Action  → (no fallback, always present)
      Thinking → Action
      Critique → Thinking → Action
      Vision   → Action (if action model is vision-capable)
      Compact  → Action
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        registry: CapabilityRegistry,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.registry = registry
        self.overrides = overrides or {}
        self._slots: dict[ModelRole, LazyLLMClient | None] = {
            ModelRole.ACTION: None,
            ModelRole.THINKING: None,
            ModelRole.CRITIQUE: None,
            ModelRole.VISION: None,
            ModelRole.COMPACT: None,
        }

    def _fallback_model(self, role: ModelRole, default_model: str) -> str:
        """Resolve which model to bind to a role, with override + fallback."""
        # Explicit config override?
        if role.value in self.overrides:
            return self.overrides[role.value]
        return default_model

    def get_client(self, role: ModelRole, default_model: str) -> LazyLLMClient:
        """Return the client for a role, lazily initializing it."""
        if self._slots[role] is not None:
            return self._slots[role]  # type: ignore[return-value]

        model = self._fallback_model(role, default_model)
        client = LazyLLMClient(self.base_url, self.api_key, model)
        self._slots[role] = client
        return client

    def thinking_client(self, default_model: str) -> LazyLLMClient:
        """Thinking → Action fallback."""
        try:
            return self.get_client(ModelRole.THINKING, default_model)
        except Exception:
            return self.get_client(ModelRole.ACTION, default_model)

    def critique_client(self, default_model: str) -> LazyLLMClient:
        """Critique → Thinking → Action fallback."""
        try:
            return self.get_client(ModelRole.CRITIQUE, default_model)
        except Exception:
            return self.thinking_client(default_model)

    def vision_client(self, default_model: str) -> LazyLLMClient:
        """Vision → Action (if vision-capable) fallback."""
        try:
            return self.get_client(ModelRole.VISION, default_model)
        except Exception:
            return self.get_client(ModelRole.ACTION, default_model)

    def compact_client(self, default_model: str) -> LazyLLMClient:
        """Compact → Action fallback."""
        try:
            return self.get_client(ModelRole.COMPACT, default_model)
        except Exception:
            return self.get_client(ModelRole.ACTION, default_model)


# ═════════════════════════════════════════════════════════════════════
#  Tool Registry — Normal Mode vs Plan Mode
# ═════════════════════════════════════════════════════════════════════

class ToolHandler(Protocol):
    """Protocol for tool call handlers."""

    name: str
    description: str
    schema: dict[str, Any]
    read_only: bool

    def execute(self, arguments: dict[str, Any]) -> Any:
        ...


class ToolRegistry:
    """Registry of available tools with mode-based filtering (Section 2.2).

    Normal Mode: all tools registered (read + write).
    Plan Mode: only read-only tools exposed.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, tool: ToolHandler) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolHandler | None:
        return self._tools.get(name)

    def list_tools(self, mode: AgentMode = AgentMode.NORMAL) -> list[ToolHandler]:
        if mode == AgentMode.PLAN:
            return [t for t in self._tools.values() if t.read_only]
        return list(self._tools.values())

    def openai_schemas(self, mode: AgentMode = AgentMode.NORMAL) -> list[dict[str, Any]]:
        tools = self.list_tools(mode)
        schemas = []
        for t in tools:
            schema = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            schemas.append(schema)
        return schemas

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry.")
        return tool.execute(arguments)


# ═════════════════════════════════════════════════════════════════════
#  Context Compaction — 5-stage staged reduction (Section 2.3.6)
# ═════════════════════════════════════════════════════════════════════

class ContextCompactor:
    """Five-stage progressive context compaction.

    Stages (triggered at increasing pressure):
      70%  → warning logged
      80%  → mask old observations (replace tool results with [masked])
      85%  → prune old tool outputs (remove oldest tool result messages)
      90%  → aggressive masking
      99%  → full LLM-based summarization via compact model
    """

    def __init__(self, registry: CapabilityRegistry, client_pool: ModelClientPool, model: str) -> None:
        self.registry = registry
        self.client_pool = client_pool
        self.model = model

    def estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total = 0
        for m in messages:
            total += len(m.get("content", "")) // 4
        return total

    def max_context(self, model: str | None = None) -> int:
        return self.registry.get_context_length(model or self.model)

    def pressure(self, messages: list[dict[str, str]], model: str | None = None) -> float:
        tokens = self.estimate_tokens(messages)
        max_ctx = self.max_context(model or self.model)
        return tokens / max_ctx

    def compact(self, messages: list[dict[str, str]], model: str | None = None) -> list[dict[str, str]]:
        """Apply progressive compaction based on pressure."""
        p = self.pressure(messages, model)

        if p > COMPACTION_FULL:
            # Stage 5: Full LLM summarization
            return self._llm_summarize(messages, model)
        elif p > COMPACTION_AGGRESSIVE_MASK:
            # Stage 4: Aggressive masking (shorten all tool results aggressively)
            return self._aggressive_mask(messages)
        elif p > COMPACTION_PRUNE:
            # Stage 3: Fast pruning (remove oldest tool result messages)
            return self._prune_tool_outputs(messages)
        elif p > COMPACTION_MASK:
            # Stage 2: Mask old observations
            return self._mask_old_observations(messages)
        elif p > COMPACTION_WARN:
            # Stage 1: Warning
            print(f"  [Context] Warning: token pressure at {p:.0%}")

        return messages

    def _mask_old_observations(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        masked = []
        for m in messages:
            if m.get("role") == "tool" or (m.get("role") == "assistant" and m.get("content", "").startswith("Observation:")):
                masked.append({"role": "system", "content": "[masked: prior observation]"})
            else:
                masked.append(m)
        return masked

    def _prune_tool_outputs(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        # Remove oldest tool result messages, keeping the most recent ones
        cleaned = []
        for m in messages:
            if m.get("role") == "tool":
                continue
            cleaned.append(m)
        return cleaned

    def _aggressive_mask(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        # Mask + truncate all non-recent messages
        result = []
        for i, m in enumerate(messages):
            if i < len(messages) - 6:  # Keep last 6 messages intact
                content = m.get("content", "")
                if len(content) > 200:
                    m = {**m, "content": content[:200] + "...[truncated]"}
            result.append(m)
        return result

    def _llm_summarize(self, messages: list[dict[str, str]], model: str | None = None) -> list[dict[str, str]]:
        # Summarize conversation using compact model (fallback: action model)
        client = self.client_pool.compact_client(model or self.model)
        context = json.dumps(messages[:min(len(messages) // 2, 20)], indent=2)
        summary_messages = [
            {"role": "system", "content": "Summarize the following conversation context into key points, preserving all facts, decisions, and current state. Be concise."},
            {"role": "user", "content": f"Summarize this context:\n\n{context}"},
        ]
        try:
            resp = client.chat(summary_messages, max_tokens=2048)
            summary = resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            summary = f"[Summary failed: {e}]"

        # Keep system prompt + summary + last few user/assistant exchanges
        preserved: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                preserved.append(m)
            if len(preserved) >= 5:
                break

        preserved.append({"role": "system", "content": f"Conversation summary: {summary}"})
        # Append last 6 messages
        preserved.extend(messages[-6:])
        return preserved


# ═════════════════════════════════════════════════════════════════════
#  Thinking Engine — explicit reasoning before action (Section 2.2.6 Phase 1)
# ═════════════════════════════════════════════════════════════════════

class ThinkingEngine:
    """Produces pre-action reasoning traces using the Thinking model role.

    Depth levels:
      OFF    → skip
      LOW    → brief reasoning (~1 paragraph)
      MEDIUM → structured reasoning (situation, plan, risks)
      HIGH   → structured reasoning + self-critique + refinement
    """

    def __init__(self, pool: ModelClientPool, registry: CapabilityRegistry) -> None:
        self.pool = pool
        self.registry = registry

    def reason(
        self,
        messages: list[dict[str, str]],
        level: ThinkingLevel = ThinkingLevel.MEDIUM,
        system_prompt: str = "",
    ) -> str:
        if level == ThinkingLevel.OFF:
            return ""

        # Use thinking model with fallback to action model
        client = self.pool.thinking_client(messages[0].get("model", "") if messages else "")

        depth_prompts = {
            ThinkingLevel.LOW: "Briefly analyze: what is the user asking, and what is one approach?",
            ThinkingLevel.MEDIUM: "Analyze the situation, outline your planned approach, and identify key risks or edge cases.",
            ThinkingLevel.HIGH: (
                "Deeply analyze the situation. Consider multiple approaches, evaluate tradeoffs, "
                "anticipate failure modes, and produce a concrete step-by-step plan."
            ),
        }

        thinking_messages = []
        if system_prompt:
            thinking_messages.append({"role": "system", "content": system_prompt})
        thinking_messages.append({
            "role": "system",
            "content": f"You are in THINKING mode. {depth_prompts[level]} Do not use tools. Output only your reasoning. No actions."
        })
        # Clone conversation context (last 20 messages) without tool calls
        for m in messages[-20:]:
            if m.get("role") != "tool":
                thinking_messages.append(m)

        try:
            resp = client.chat(thinking_messages, temperature=0.5, max_tokens=4096)
            trace = resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            return f"[Thinking failed: {e}]"

        # HIGH level: self-critique + refinement
        if level == ThinkingLevel.HIGH and trace:
            critique = self._critique(trace, messages, system_prompt)
            refined = self._refine(trace, critique, messages, system_prompt)
            if refined:
                trace = refined

        return trace

    def _critique(self, trace: str, messages: list[dict[str, str]], system_prompt: str) -> str:
        client = self.pool.critique_client(messages[0].get("model", "") if messages else "")
        critique_messages = []
        if system_prompt:
            critique_messages.append({"role": "system", "content": system_prompt})
        critique_messages.append({
            "role": "system",
            "content": "You are in CRITIQUE mode. Review the following reasoning trace and identify: (1) logical flaws, (2) missed edge cases, (3) better alternatives. Be concise. No actions."
        })
        critique_messages.append({"role": "user", "content": f"Trace to critique:\n\n{trace}"})
        try:
            resp = client.chat(critique_messages, temperature=0.5, max_tokens=2048)
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            return f"[Critique failed: {e}]"

    def _refine(self, trace: str, critique: str, messages: list[dict[str, str]], system_prompt: str) -> str:
        client = self.pool.thinking_client(messages[0].get("model", "") if messages else "")
        refine_messages = []
        if system_prompt:
            refine_messages.append({"role": "system", "content": system_prompt})
        refine_messages.append({
            "role": "system",
            "content": "You are in REFINEMENT mode. Incorporate the critique into the original reasoning trace to produce an improved version. No actions."
        })
        refine_messages.append({
            "role": "user",
            "content": f"Original trace:\n{trace}\n\nCritique:\n{critique}\n\nProduce the refined trace."
        })
        try:
            resp = client.chat(refine_messages, temperature=0.5, max_tokens=4096)
            return resp.choices[0].message.content if resp.choices else trace
        except Exception:
            return trace


# ═════════════════════════════════════════════════════════════════════
#  Doom-Loop Detection (Section 2.2.6 Phase 3)
# ═════════════════════════════════════════════════════════════════════

class DoomLoopDetector:
    """Fingerprint-based detection for repeated identical tool calls.

    Tracks MD5(tool_name + sorted args) in a sliding window.
    Tier 1 (3 repetitions): inject [SYSTEM WARNING] into conversation.
    Tier 2 (after warning, same fingerprint recurs): pause via ApprovalManager.
    """

    def __init__(self, window_size: int = DOOM_WINDOW_SIZE, threshold: int = DOOM_REPEAT_THRESHOLD) -> None:
        self.fingerprints: deque[str] = deque(maxlen=window_size)
        self.threshold = threshold
        self._warned: set[str] = set()  # Fingerprints that have triggered a warning
        self._paused: set[str] = set()  # Fingerprints that have been paused
        self._one_shot_approved: set[str] = set()  # Fingerprints allowed once after pause

    def _fingerprint(self, tool_name: str, arguments: dict[str, Any]) -> str:
        args_str = json.dumps(arguments, sort_keys=True)
        return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()

    def check(self, tool_calls: list[dict[str, Any]]) -> tuple[bool, str]:
        """Returns (should_execute, reason) where should_execute is False if doom loop detected."""
        for tc in tool_calls:
            fp = self._fingerprint(tc.get("name", ""), tc.get("arguments", {}))
            self.fingerprints.append(fp)

        counts = Counter(self.fingerprints)
        for fp, count in counts.items():
            if count >= self.threshold:
                if fp in self._paused:
                    # Tier 2: already paused once, need re-approval
                    return False, f"DOOM LOOP (Tier 2): repeated tool call detected after pause. Fingerprint: {fp[:8]}..."
                elif fp in self._warned:
                    # Escalate to pause
                    self._paused.add(fp)
                    return False, f"DOOM LOOP (Tier 2): repeated tool call after warning. Fingerprint: {fp[:8]}..."
                else:
                    # Tier 1: first warning
                    self._warned.add(fp)
                    return True, f"DOOM LOOP (Tier 1): repeated tool call detected. Fingerprint: {fp[:8]}..."

        return True, ""

    def get_warning_message(self, fingerprint_short: str) -> str:
        return f"[SYSTEM WARNING] The agent has called the same tool with the same arguments repeatedly. Try a different approach."

    def reset(self) -> None:
        self.fingerprints.clear()
        self._warned.clear()
        self._paused.clear()
        self._one_shot_approved.clear()


# ═════════════════════════════════════════════════════════════════════
#  Extended ReAct Executor (Section 2.2.6)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class IterationContext:
    """Per-iteration state container."""
    iteration: int = 0
    nudge_count: int = 0
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    terminated: bool = False
    error: str | None = None


class ReActExecutor:
    """Extended ReAct loop with five-phase execution.

    Phase 0: Staged context management (compaction)
    Phase 1: Thinking (optional, at configurable depth)
    Phase 2: Action (tool call with action model)
    Phase 3: Decision, dispatch, doom-loop detection, observation
    """

    def __init__(
        self,
        config: "ConfigManager",
        session: "SessionManager",
        approval: "ApprovalManager",
        pool: ModelClientPool,
        registry: CapabilityRegistry,
        tool_registry: "ToolRegistry",
        thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM,
        context_engine: Any = None,
        tool_context: Any = None,
    ) -> None:
        self.config = config
        self.session = session
        self.approval = approval
        self.pool = pool
        self.registry = registry
        self.tool_registry = tool_registry
        self.thinking_engine = ThinkingEngine(pool, registry)
        self.compactor = ContextCompactor(registry, pool, config.model)
        self.doom_detector = DoomLoopDetector()
        self.thinking_level = thinking_level
        self.iteration_ctx = IterationContext()
        # Context Engineering Layer (PromptComposer, Reminders, Memory,
        # Compaction) and the Tool/Context service bundle. Optional so the
        # executor still works standalone, but AgenticAgent always wires them.
        self.context_engine = context_engine
        self.tool_context = tool_context
        # If a ContextEngine is present, share its compactor so token-pressure
        # accounting is consistent across the whole layer.
        if context_engine is not None and getattr(context_engine, "compactor", None) is not None:
            self.compactor = context_engine.compactor
        self._current_user_message: str = ""
        self._consecutive_reads: int = 0

    def _runtime_context(self) -> dict[str, Any]:
        """Build the environment snapshot for conditional prompt composition."""
        import os as _os
        cwd = _os.getcwd()
        skills_index = ""
        if self.tool_context is not None and getattr(self.tool_context, "skill_loader", None):
            skills_index = self.tool_context.skill_loader.index_text()
        return {
            "cwd": cwd,
            "in_git_repo": (Path(cwd) / ".git").exists(),
            "task_tracking_enabled": True,
            "skills_index": skills_index,
        }

    def _system_prompt(self) -> str:
        """Composed system prompt (PromptComposer) with fallback to raw config."""
        if self.context_engine is not None:
            try:
                return self.context_engine.build_system_prompt(self._runtime_context())
            except Exception:
                pass
        return self.config.system_prompt

    def run(self, user_message: str) -> str:
        """Process a user message through the full ReAct loop."""
        self.session.add("user", user_message)
        self.iteration_ctx = IterationContext()
        self._current_user_message = user_message
        self._consecutive_reads = 0

        # Surface relevant cross-session memory bullets as a preamble (ACE).
        if self.context_engine is not None:
            try:
                preamble = self.context_engine.memory_preamble(user_message)
                if preamble:
                    self.session.add("system", preamble)
            except Exception:
                pass

        while not self.iteration_ctx.terminated and self.iteration_ctx.iteration < self.iteration_ctx.max_iterations:
            self._run_iteration()

        # Reflect on the outcome for cross-session learning.
        final = ""
        if self.session.messages and self.session.messages[-1].role == "assistant":
            final = self.session.messages[-1].content
        if self.context_engine is not None:
            try:
                success = "[Error" not in final and "[ACTION PAUSED]" not in final
                self.context_engine.reflect(user_message, final, success)
            except Exception:
                pass
        return final

    def _run_iteration(self) -> None:
        ctx = self.iteration_ctx
        ctx.iteration += 1

        messages = self.session.to_openai_messages(self._system_prompt())

        # ═══════════════════════════════════════════════════════════════
        # Phase 0: Staged context management
        # ═══════════════════════════════════════════════════════════════
        messages = self.compactor.compact(messages)

        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Thinking (optional)
        # ═══════════════════════════════════════════════════════════════
        if self.thinking_level != ThinkingLevel.OFF:
            trace = self.thinking_engine.reason(messages, self.thinking_level, self.config.system_prompt)
            if trace:
                # Inject reasoning trace as system reminder
                messages.append({
                    "role": "system",
                    "content": f"[THINKING TRACE]\n{trace}\n[END THINKING TRACE]"
                })

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Action — call action model with tools
        # ═══════════════════════════════════════════════════════════════
        action_client = self.pool.get_client(ModelRole.ACTION, self.config.model)
        tool_schemas = self.tool_registry.openai_schemas(AgentMode.NORMAL)

        try:
            if tool_schemas:
                resp = action_client.chat(messages, tools=tool_schemas, tool_choice="auto")
            else:
                resp = action_client.chat(messages)
        except Exception as e:
            self.session.add("assistant", f"[Error during action phase: {e}]")
            ctx.terminated = True
            return

        message = resp.choices[0].message if resp.choices else None
        if message is None:
            ctx.terminated = True
            return

        content = message.content or ""
        tool_calls_raw = getattr(message, "tool_calls", None)

        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Decision, Doom-Loop Detection, Dispatch
        # ═══════════════════════════════════════════════════════════════
        if tool_calls_raw:
            # Parse tool calls
            tool_calls = []
            for tc in tool_calls_raw:
                tool_calls.append({
                    "id": tc.id if hasattr(tc, "id") else str(hash(tc)),
                    "name": tc.function.name if hasattr(tc, "function") else "",
                    "arguments": json.loads(tc.function.arguments) if (hasattr(tc, "function") and tc.function.arguments) else {},
                })

            # Doom-loop detection
            should_execute, reason = self.doom_detector.check(tool_calls)
            if not should_execute:
                # Tier 2: pause
                approved = self.approval.request("doom_loop", f"Agent repeating same action: {reason}. Allow / Break?")
                if not approved:
                    self.session.add("assistant", "[ACTION PAUSED] Agent is in a loop. Please provide guidance.")
                    ctx.terminated = True
                    return
                else:
                    # One-shot: allow this execution and continue
                    pass
            elif reason:  # Tier 1: warning was injected
                warning_msg = self.doom_detector.get_warning_message(reason.split()[-1])
                self.session.add("system", warning_msg)

            # Approval gate for each tool call
            results = []
            for tc in tool_calls:
                tool_name = tc["name"]
                args = tc["arguments"]
                desc = f"{tool_name}({json.dumps(args)})"

                if self.approval.needs_approval(tool_name, desc):
                    if not self.approval.request(tool_name, desc):
                        results.append({"tool": tool_name, "result": "[REJECTED by user]"})
                        continue

                try:
                    result = self.tool_registry.execute(tool_name, args)
                except Exception as e:
                    result = f"[Error: {e}]"

                results.append({"tool": tool_name, "result": result})

            # Build observation message and add to session
            obs_parts = []
            any_read = False
            for res in results:
                obs_parts.append(f"Tool: {res['tool']}\nResult: {res['result']}")
                # Track read-only tool calls for exploration-spiral detection.
                handler = self.tool_registry.get(res["tool"])
                if handler is not None and getattr(handler, "read_only", False):
                    any_read = True
            if any_read:
                self._consecutive_reads += 1
            else:
                self._consecutive_reads = 0
            observation = "\n\n".join(obs_parts)
            self.session.add("assistant", f"[Tool execution results]\n{observation}\nUser-facing response: {content}")
        else:
            # No tool calls — text-only response
            self._consecutive_reads = 0
            self.session.add("assistant", content)
            ctx.terminated = True

        # ═══════════════════════════════════════════════════════════════
        # Phase 3b: Event-driven system reminders (Section 2.3.4)
        # ═══════════════════════════════════════════════════════════════
        self._inject_reminders()

        # Nudge on repeated failures
        if ctx.nudge_count < NUDGE_BUDGET and self._last_tool_failed():
            nudge = self._smart_nudge()
            self.session.add("system", nudge)
            ctx.nudge_count += 1

    def _inject_reminders(self) -> None:
        """Run the eight event detectors and inject any firing reminders.

        Reminders are placed at maximum recency (appended last) so the next
        LLM call sees them as the most recent input (Section 2.3.4).
        """
        if self.context_engine is None:
            return
        try:
            from tools_plus_context import ConversationState
        except Exception:
            return

        state = ConversationState(
            last_tool_failed=self._last_tool_failed(),
            retried_after_failure=False,
            consecutive_reads=self._consecutive_reads,
            signaled_completion=self.iteration_ctx.terminated,
            empty_completion_message=(
                self.iteration_ctx.terminated
                and bool(self.session.messages)
                and not self.session.messages[-1].content.strip()
            ),
        )
        for reminder in self.context_engine.reminders_for(state):
            self.session.add("user", reminder)

    def _last_tool_failed(self) -> bool:
        # Check if the most recent tool result was an error
        for msg in reversed(self.session.messages):
            if msg.role == "assistant" and "[Error:" in msg.content:
                return True
            if msg.role == "tool" and "[Error:" in msg.content:
                return True
        return False

    def _smart_nudge(self) -> str:
        # Analyze last error and produce targeted guidance
        for msg in reversed(self.session.messages):
            if msg.role == "assistant" and "[Error:" in msg.content:
                if "not found" in msg.content:
                    return "[SYSTEM NUDGE] The previous tool reported 'not found'. Verify the path or identifier and retry with a corrected argument."
                elif "permission" in msg.content.lower():
                    return "[SYSTEM NUDGE] Permission was denied. Check file permissions or use an alternative approach."
                else:
                    return "[SYSTEM NUDGE] The previous action failed. Reassess your approach and try a different strategy."
        return "[SYSTEM NUDGE] The previous action failed. Try an alternative approach."


# ═════════════════════════════════════════════════════════════════════
#  AgenticAgent — the agentic mode driver, injected with 4 managers
# ═════════════════════════════════════════════════════════════════════

class AgenticAgent:
    """The agentic-mode agent that wires the Extended ReAct loop.

    Receives the same four managers as RiftAgent via injection.
    Integrates model role pool, tool registry, and the ReAct executor.
    """

    def __init__(
        self,
        config: "ConfigManager",
        session: "SessionManager",
        mode: "ModeManager",
        approval: "ApprovalManager",
        model_overrides: dict[str, str] | None = None,
        thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM,
    ) -> None:
        self.config = config
        self.session = session
        self.mode = mode
        self.approval = approval

        # Capability registry (local cache)
        self.registry = CapabilityRegistry()
        self.registry.bootstrap_defaults()

        # Model client pool (5 roles, lazy init)
        self.pool = ModelClientPool(
            base_url=config.base_url,
            api_key=config.api_key,
            registry=self.registry,
            overrides=model_overrides or {},
        )

        # Tool registry (Normal + Plan modes)
        self.tool_registry = ToolRegistry()
        self._register_builtin_tools()

        # ── Tool & Context Engineering Layer (tools_plus_context.py) ──────
        # Lazy import to avoid a circular dependency: tools_plus_context
        # imports primitives from this module.
        try:
            from tools_plus_context import (
                build_context_engine,
                install_tool_and_context_layer,
            )
            self.tool_context = install_tool_and_context_layer(self)
            self.compactor = ContextCompactor(self.registry, self.pool, config.model)
            self.context_engine = build_context_engine(config, self.compactor)
        except Exception as e:
            print(f"  [Warn] Tool/Context layer unavailable: {e}")
            self.tool_context = None
            self.context_engine = None

        # ReAct executor (wired with the context engine + tool context)
        self.executor = ReActExecutor(
            config=config,
            session=session,
            approval=approval,
            pool=self.pool,
            registry=self.registry,
            tool_registry=self.tool_registry,
            thinking_level=thinking_level,
            context_engine=self.context_engine,
            tool_context=self.tool_context,
        )

        self._client: Any = None

    def _register_builtin_tools(self) -> None:
        """Register built-in read-only and read-write tools."""
        # Read-only tools
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(ListDirectoryTool())
        self.tool_registry.register(SearchFilesTool())
        # Read-write tools
        self.tool_registry.register(WriteFileTool())
        self.tool_registry.register(ShellExecTool())
        self.tool_registry.register(WebFetchTool())

    @property
    def client(self):
        """Lazily initialize the OpenAI-compatible client (action model)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                print("Error: 'openai' package not found.\nInstall with: /usr/local/bin/pip3 install openai")
                import sys
                sys.exit(1)
            self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return self._client

    def chat(self, user_input: str) -> str:
        """Process user input through the Extended ReAct loop."""
        return self.executor.run(user_input)

    def run_interactive(self) -> None:
        """Multi-turn REPL loop in agentic mode."""
        print("\n  Rift agentic mode. Type 'exit' or Ctrl+C to quit.\n")
        while True:
            try:
                user_input = input("  you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye.\n")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", ":q"):
                print("\n  Goodbye.\n")
                break

            reply = self.chat(user_input)
            print(f"\n  rift > {reply}\n")


# ═════════════════════════════════════════════════════════════════════
#  Built-in Tool Implementations
# ═════════════════════════════════════════════════════════════════════

class ReadFileTool:
    name = "read_file"
    description = "Read the contents of a file."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file"},
        },
        "required": ["path"],
    }
    read_only = True

    def execute(self, arguments: dict[str, Any]) -> Any:
        path = Path(arguments["path"])
        if not path.exists():
            return f"[Error: File not found: {path}]"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"[Error: {e}]"


class ListDirectoryTool:
    name = "list_directory"
    description = "List all files and directories in a given path."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the directory"},
        },
        "required": ["path"],
    }
    read_only = True

    def execute(self, arguments: dict[str, Any]) -> Any:
        path = Path(arguments["path"])
        if not path.exists():
            return f"[Error: Directory not found: {path}]"
        try:
            items = []
            for item in sorted(path.iterdir()):
                items.append(f"{'[DIR] ' if item.is_dir() else '[FILE]'} {item.name}")
            return "\n".join(items)
        except Exception as e:
            return f"[Error: {e}]"


class SearchFilesTool:
    name = "search_files"
    description = "Search for files matching a pattern in a directory."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to search in"},
            "pattern": {"type": "string", "description": "Filename pattern (glob)"},
        },
        "required": ["path", "pattern"],
    }
    read_only = True

    def execute(self, arguments: dict[str, Any]) -> Any:
        import fnmatch
        path = Path(arguments["path"])
        pattern = arguments["pattern"]
        matches = []
        for root, _dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, pattern):
                matches.append(str(Path(root) / filename))
        return "\n".join(matches) if matches else "[No matches found]"


class WriteFileTool:
    name = "write_file"
    description = "Write content to a file (creates or overwrites)."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }
    read_only = False

    def execute(self, arguments: dict[str, Any]) -> Any:
        path = Path(arguments["path"])
        try:
            path.write_text(arguments["content"], encoding="utf-8")
            return f"[Written {len(arguments['content'])} bytes to {path}]"
        except Exception as e:
            return f"[Error: {e}]"


class ShellExecTool:
    name = "shell_exec"
    description = "Execute a shell command and return stdout + stderr."
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
        },
        "required": ["command"],
    }
    read_only = False

    def execute(self, arguments: dict[str, Any]) -> Any:
        import subprocess
        command = arguments["command"]
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            out = f"EXIT CODE: {result.returncode}\n"
            out += f"STDOUT:\n{result.stdout}\n"
            out += f"STDERR:\n{result.stderr}\n"
            return out
        except subprocess.TimeoutExpired:
            return "[Error: Command timed out after 60s]"
        except Exception as e:
            return f"[Error: {e}]"


class WebFetchTool:
    name = "web_fetch"
    description = "Fetch the content of a web page."
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
        },
        "required": ["url"],
    }
    read_only = False

    def execute(self, arguments: dict[str, Any]) -> Any:
        import urllib.request
        url = arguments["url"]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Rift-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"[Error: {e}]"
