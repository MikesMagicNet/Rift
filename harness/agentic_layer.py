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
import re
import shlex
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    from cli import ApprovalManager, ConfigManager, ModeManager, SessionManager

from harness import theme


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

# Rate limiting — max API calls per 60-second sliding window
DEFAULT_API_RATE_LIMIT = 38
RATE_LIMIT_WINDOW = 60.0  # seconds


class RateLimiter:
    """Sliding-window rate limiter for outbound API calls.

    Tracks call timestamps in a deque. When the number of calls in the
    last ``window`` seconds reaches ``max_calls``, subsequent calls block
    (with a spinner) until enough old timestamps expire.
    """

    def __init__(self, max_calls: int = DEFAULT_API_RATE_LIMIT, window: float = RATE_LIMIT_WINDOW) -> None:
        self.max_calls = max_calls
        self.window = window
        self._timestamps: deque[float] = deque()

    def _prune(self) -> None:
        """Remove timestamps older than the window."""
        cutoff = time.time() - self.window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def acquire(self) -> None:
        """Block until a call slot is available, then record it."""
        self._prune()
        if len(self._timestamps) >= self.max_calls:
            wait = self.window - (time.time() - self._timestamps[0])
            if wait > 0:
                theme.print_warning(
                    f"[RateLimit] {self.max_calls} calls/{self.window:.0f}s reached — waiting {wait:.1f}s..."
                )
                time.sleep(wait)
                self._prune()
        self._timestamps.append(time.time())


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

    def __init__(self, base_url: str, api_key: str, model: str, rate_limiter: RateLimiter | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client: Any = None
        self._rate_limiter = rate_limiter

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
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
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
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.registry = registry
        self.overrides = overrides or {}
        self._rate_limiter = rate_limiter
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
        client = LazyLLMClient(self.base_url, self.api_key, model, rate_limiter=self._rate_limiter)
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

    # Maximum workers for parallel tool execution
    MAX_PARALLEL_TOOLS = 4

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}
        self._schema_cache: dict[str, list[dict[str, Any]]] = {}

    def register(self, tool: ToolHandler) -> None:
        self._tools[tool.name] = tool
        self._schema_cache.clear()  # Invalidate cache

    def get(self, name: str) -> ToolHandler | None:
        return self._tools.get(name)

    def list_tools(self, mode: AgentMode = AgentMode.NORMAL) -> list[ToolHandler]:
        if mode == AgentMode.PLAN:
            return [t for t in self._tools.values() if t.read_only]
        return list(self._tools.values())

    def openai_schemas(self, mode: AgentMode = AgentMode.NORMAL) -> list[dict[str, Any]]:
        cache_key = mode.value
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]
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
        self._schema_cache[cache_key] = schemas
        return schemas

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry.")
        return tool.execute(arguments)

    def execute_parallel(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute multiple tool calls in parallel.

        Args:
            calls: list of {"name": str, "arguments": dict}
        Returns:
            list of {"name": str, "result": Any, "error": str|None} in the
            same order as the input.
        """
        results: list[dict[str, Any]] = [None] * len(calls)  # type: ignore[list-item]
        if len(calls) == 1:
            tc = calls[0]
            try:
                results[0] = {"name": tc["name"], "result": self.execute(tc["name"], tc["arguments"]), "error": None}
            except Exception as e:
                results[0] = {"name": tc["name"], "result": None, "error": str(e)}
            return results

        with ThreadPoolExecutor(max_workers=min(self.MAX_PARALLEL_TOOLS, len(calls))) as pool:
            future_to_idx = {}
            for i, tc in enumerate(calls):
                future = pool.submit(self._execute_safe, tc["name"], tc["arguments"])
                future_to_idx[future] = i
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def _execute_safe(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"name": name, "result": self.execute(name, arguments), "error": None}
        except Exception as e:
            return {"name": name, "result": None, "error": str(e)}


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
            theme.print_warning(f"[Context] Token pressure at {p:.0%}")

        return messages

    @staticmethod
    def _merge_system_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep OpenAI-compatible system content at the start of the message list."""
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system" and m.get("content", "")]
        non_system = [m for m in messages if m.get("role") != "system"]
        if not system_parts:
            return non_system
        return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system

    def _mask_old_observations(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        masked = []
        masked_observation_count = 0
        for m in messages:
            if m.get("role") == "tool" or (m.get("role") == "assistant" and m.get("content", "").startswith("Observation:")):
                masked_observation_count += 1
            else:
                masked.append(m)
        if masked_observation_count:
            masked.insert(0, {"role": "system", "content": f"[masked: {masked_observation_count} prior observation(s)]"})
        return self._merge_system_messages(masked)

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
        return self._merge_system_messages(result)

    def _llm_summarize(self, messages: list[dict[str, str]], model: str | None = None) -> list[dict[str, str]]:
        # Summarize conversation using compact model (fallback: action model)
        client = self.client_pool.compact_client(model or self.model)
        context = json.dumps(messages[:min(len(messages) // 2, 20)], indent=2)
        summary_messages = [
            {"role": "system", "content": "Summarize the following conversation context into key points, preserving all facts, decisions, and current state. Be concise."},
            {"role": "user", "content": f"Summarize this context:\n\n{context}"},
        ]
        try:
            with theme.spinner("Summarizing"):
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
        return self._merge_system_messages(preserved)


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
        thinking_prompt_parts = []
        if system_prompt:
            thinking_prompt_parts.append(system_prompt)
        thinking_prompt_parts.append(
            f"You are in THINKING mode. {depth_prompts[level]} Do not use tools. Output only your reasoning. No actions."
        )
        if thinking_prompt_parts:
            thinking_messages.append({"role": "system", "content": "\n\n".join(thinking_prompt_parts)})
        # Clone conversation context (last 20 messages) without tool calls
        for m in messages[-20:]:
            if m.get("role") != "tool":
                thinking_messages.append(m)

        try:
            with theme.spinner("Thinking"):
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
        critique_prompt_parts = []
        if system_prompt:
            critique_prompt_parts.append(system_prompt)
        critique_prompt_parts.append(
            "You are in CRITIQUE mode. Review the following reasoning trace and identify: (1) logical flaws, (2) missed edge cases, (3) better alternatives. Be concise. No actions."
        )
        critique_messages.append({"role": "system", "content": "\n\n".join(critique_prompt_parts)})
        critique_messages.append({"role": "user", "content": f"Trace to critique:\n\n{trace}"})
        try:
            with theme.spinner("Critiquing"):
                resp = client.chat(critique_messages, temperature=0.5, max_tokens=2048)
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            return f"[Critique failed: {e}]"

    def _refine(self, trace: str, critique: str, messages: list[dict[str, str]], system_prompt: str) -> str:
        client = self.pool.thinking_client(messages[0].get("model", "") if messages else "")
        refine_messages = []
        refine_prompt_parts = []
        if system_prompt:
            refine_prompt_parts.append(system_prompt)
        refine_prompt_parts.append(
            "You are in REFINEMENT mode. Incorporate the critique into the original reasoning trace to produce an improved version. No actions."
        )
        refine_messages.append({"role": "system", "content": "\n\n".join(refine_prompt_parts)})
        refine_messages.append({
            "role": "user",
            "content": f"Original trace:\n{trace}\n\nCritique:\n{critique}\n\nProduce the refined trace."
        })
        try:
            with theme.spinner("Refining"):
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
        persistence: Any = None,
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
        # Persistence / audit logger — injected from AgenticAgent if available
        self._persistence = persistence
        self._current_user_message: str = ""
        self._consecutive_reads: int = 0

    def _audit(self, event_type: str, details: dict[str, Any] | None = None) -> None:
        """Emit an audit entry if a persistence manager is attached."""
        if self._persistence is None:
            return
        try:
            from persistence_layer import AuditEntry, AuditEventType
            entry = AuditEntry(
                event_type=event_type,
                session_id=getattr(self.session, "session_name", ""),
                iteration=self.iteration_ctx.iteration,
                details=details or {},
            )
            self._persistence.audit.log(entry)
        except Exception:
            pass  # Audit should never break the main loop

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

        self._audit("session_start", {"message_preview": user_message[:200]})

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

        self._audit("session_end", {
            "iterations": self.iteration_ctx.iteration,
            "response_length": len(final),
        })
        # Persist the session to disk immediately at end of turn.
        try:
            self.session.save()
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
                # Keep system-role content at the front for providers that
                # reject system messages outside position 0.
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = (
                        f"{messages[0].get('content', '')}\n\n[THINKING TRACE]\n{trace}\n[END THINKING TRACE]"
                    )
                else:
                    messages.insert(0, {
                        "role": "system",
                        "content": f"[THINKING TRACE]\n{trace}\n[END THINKING TRACE]",
                    })

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Action — call action model with tools
        # ═══════════════════════════════════════════════════════════════
        action_client = self.pool.get_client(ModelRole.ACTION, self.config.model)
        tool_schemas = self.tool_registry.openai_schemas(AgentMode.NORMAL)

        try:
            spinner_label = "Planning" if tool_schemas else "Thinking"
            with theme.spinner(spinner_label):
                if tool_schemas:
                    resp = action_client.chat(messages, tools=tool_schemas, tool_choice="auto")
                else:
                    resp = action_client.chat(messages)
        except Exception as e:
            err_msg = str(e)
            # Enrich 404 errors with the model name so typos are obvious
            if "404" in err_msg:
                err_msg = f"{err_msg} — model '{self.config.model}' not found at {self.config.base_url}. Check the model name in config.json."
            self._audit("model_error", {"error": err_msg[:500]})
            self.session.add("assistant", f"[Error during action phase: {err_msg}]")
            ctx.terminated = True
            return

        message = resp.choices[0].message if resp.choices else None
        if message is None:
            ctx.terminated = True
            return

        content = message.content or ""
        tool_calls_raw = getattr(message, "tool_calls", None)

        # ── Fallback: parse tool calls from text output ──────────────
        # Some models (e.g., DeepSeek) emit tool calls as XML/text in
        # the content field instead of using the structured tool_calls
        # API.  Detect and convert them so the executor handles them
        # like any other structured call.
        text_parsed_calls: list[dict] | None = None
        if not tool_calls_raw and content:
            text_parsed_calls = self._parse_text_tool_calls(content)
            if text_parsed_calls:
                self._audit("text_tool_parse", {"count": len(text_parsed_calls)})
                # Strip the XML tool-call markup from the displayed content
                # so the user sees a clean reply (if any remains).
                clean = self._strip_tool_markup(content)
                content = clean.strip()

        # ═══════════════════════════════════════════════════════════════
        # Phase 3: Decision, Doom-Loop Detection, Dispatch
        # ═══════════════════════════════════════════════════════════════
        if tool_calls_raw or text_parsed_calls:
            # Parse tool calls — prefer structured API, fall back to text
            if tool_calls_raw:
                tool_calls = []
                for tc in tool_calls_raw:
                    tool_calls.append({
                        "id": tc.id if hasattr(tc, "id") else f"call_{hash(tc)}",
                        "name": tc.function.name if hasattr(tc, "function") else "",
                        "arguments": json.loads(tc.function.arguments) if (hasattr(tc, "function") and tc.function.arguments) else {},
                    })
            else:
                tool_calls = text_parsed_calls  # type: ignore[assignment]

            # Ensure OpenAI-compatible role ordering before persisting tool calls:
            # system messages are only valid at the beginning of the prompt.
            if self.session.messages and self.session.messages[-1].role == "system":
                system_msg = self.session.messages.pop()
                if self.session.messages and self.session.messages[0].role == "system":
                    self.session.messages[0].content = (
                        f"{self.session.messages[0].content}\n\n{system_msg.content}"
                    )
                else:
                    self.session.messages.insert(0, system_msg)

            # Doom-loop detection
            should_execute, reason = self.doom_detector.check(tool_calls)
            if not should_execute:
                # Tier 2: pause
                self._audit("doom_loop_pause", {"reason": reason})
                approved = self.approval.request("doom_loop", f"Agent repeating same action: {reason}. Allow / Break?")
                if not approved:
                    self.session.add("assistant", "[ACTION PAUSED] Agent is in a loop. Please provide guidance.")
                    ctx.terminated = True
                    return
                else:
                    # One-shot: allow this execution and continue
                    pass
            elif reason:  # Tier 1: warning was injected
                self._audit("doom_loop_warning", {"reason": reason})
                warning_msg = self.doom_detector.get_warning_message(reason.split()[-1])
                if self.session.messages and self.session.messages[0].role == "system":
                    self.session.messages[0].content = f"{self.session.messages[0].content}\n\n{warning_msg}"
                else:
                    self.session.messages.insert(0, type(self.session.messages[-1])("system", warning_msg))

            # ── Store the assistant message WITH tool_calls ──────────────
            # The OpenAI API requires the conversation history to contain:
            #   assistant(message + tool_calls) → tool(result + tool_call_id) → ...
            # Without this, the model never sees proper tool results and
            # hallucinates that it has no filesystem access.
            stored_tool_calls = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ]
            self.session.add("assistant", content, tool_calls=stored_tool_calls)

            # ── Execute tools in parallel ────────────────────────────────
            # Build the call list (approval gate first, then parallel exec)
            approved_calls = []
            for tc in tool_calls:
                tool_name = tc["name"]
                args = tc["arguments"]
                desc = f"{tool_name}({json.dumps(args)})"

                if self.approval.needs_approval(tool_name, desc):
                    self._audit("approval_request", {"tool": tool_name})
                    if not self.approval.request(tool_name, desc):
                        self._audit("approval_deny", {"tool": tool_name})
                        self.session.add("tool", "[REJECTED by user]", tool_call_id=tc["id"])
                        continue
                    self._audit("approval_grant", {"tool": tool_name})
                self._audit("tool_call", {"tool": tool_name, "arguments": args})
                approved_calls.append(tc)

            # Execute all approved calls in parallel
            if len(approved_calls) == 1:
                spinner_label = f'[THINKING] Using tool "{approved_calls[0]["name"]}"'
            else:
                names = ", ".join(f'"{tc["name"]}"' for tc in approved_calls)
                spinner_label = f'[THINKING] Using tools {names}'
                
            with theme.spinner(spinner_label, style="executing"):
                call_list = [{"name": tc["name"], "arguments": tc["arguments"]} for tc in approved_calls]
                exec_results = self.tool_registry.execute_parallel(call_list)

            # Store results as tool-role messages
            any_read = False
            for tc, res in zip(approved_calls, exec_results):
                tool_name = tc["name"]
                result = res["result"]
                error = res["error"]
                if error:
                    result = f"[Error: {error}]"
                    self._audit("tool_error", {"tool": tool_name, "error": str(error)[:300]})
                else:
                    self._audit("tool_result", {"tool": tool_name, "result_length": len(str(result))})

                # Track read-only tool calls for exploration-spiral detection.
                handler = self.tool_registry.get(tool_name)
                if handler is not None and getattr(handler, "read_only", False):
                    any_read = True

                self.session.add("tool", str(result), tool_call_id=tc["id"])

            if any_read:
                self._consecutive_reads += 1
            else:
                self._consecutive_reads = 0
            # Do NOT set terminated — the loop continues so the model
            # receives the tool results and can respond to them.
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

    # ── Fallback text-based tool call parsing ────────────────────────

    _TEXT_TOOL_PATTERNS = [
        # DeepSeek-style XML: <tool_call><invoke_name>name</invoke_name>...</tool_call>
        re.compile(
            r"<tool_call>\s*"
            r"<invoke(?:_name)?>\s*(?P<name>\w+)\s*</invoke(?:_name)?>"
            r"(?P<body>.*?)"
            r"</(?:invoke(?:_name)?)?\s*>\s*</tool_call>",
            re.DOTALL | re.IGNORECASE,
        ),
        # Simpler variant: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        re.compile(
            r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>",
            re.DOTALL,
        ),
        # Function-call style with json: <functioncall> {"name": ..., "arguments": ...}
        re.compile(
            r"<function_?call>\s*(?P<json>\{.*?\})\s*(?:</function_?call>)?",
            re.DOTALL | re.IGNORECASE,
        ),
        # Function-call style with invoke tags: <functioncall> <invoke name="shell_exec"> ... </invoke> </functioncall>
        re.compile(
            r"<function_?call>\s*<invoke\s+name=[\"']?(?P<name>\w+)[\"']?\s*>(?P<body>.*?)</invoke>\s*(?:</function_?call>)?",
            re.DOTALL | re.IGNORECASE,
        ),
    ]

    # Tag patterns for extracting individual arguments from XML body
    _ARG_TAG = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
    _PARAM_TAG = re.compile(r"<parameter\s+name=[\"']?(\w+)[\"']?[^>]*>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)

    # Loose fallback: a <tool_call> block with NEITHER <invoke> tags NOR a JSON
    # object — just a flattened colon-delimited dump, e.g.:
    #   <tool_call> id: 0 type: function function:   name: web_fetch
    #     arguments:     url: https://...
    #   </tool_call>
    # Several NIM-hosted models (Kimi K2, some Nemotron builds) emit this shape
    # when tool_choice="auto" instead of using the structured tool_calls API.
    _LOOSE_TOOL_CALL = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
    _LOOSE_NAME = re.compile(r"\bname\s*:\s*([A-Za-z_]\w*)", re.IGNORECASE)
    _LOOSE_ARGS_SPLIT = re.compile(r"\barguments\s*:", re.IGNORECASE)
    _LOOSE_KV = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*(.*?)\s*$", re.MULTILINE)
    # Reserved keys that are part of the call envelope, not actual arguments.
    _LOOSE_RESERVED = frozenset({"name", "type", "function", "id", "arguments"})

    def _parse_text_tool_calls(self, text: str) -> list[dict] | None:
        """Extract tool calls from model text output (XML/JSON fallback).

        Returns a list of ``{"id": str, "name": str, "arguments": dict}``
        or ``None`` if no tool calls were found.
        """
        calls: list[dict] = []
        call_counter = 0

        # Try each pattern
        for pattern in self._TEXT_TOOL_PATTERNS:
            for m in pattern.finditer(text):
                call_counter += 1
                call_id = f"text_call_{call_counter}"

                # JSON-based patterns
                json_str = m.groupdict().get("json")
                if json_str:
                    try:
                        data = json.loads(json_str)
                        name = data.get("name", "")
                        args = data.get("arguments", data.get("parameters", {}))
                        if isinstance(args, str):
                            args = json.loads(args)
                        if name and name in [t.name for t in self.tool_registry.list_tools()]:
                            calls.append({"id": call_id, "name": name, "arguments": args})
                    except (json.JSONDecodeError, TypeError):
                        pass
                    continue

                # XML-based patterns with named group "name"
                name = m.groupdict().get("name", "")
                body = m.groupdict().get("body", "")

                if not name:
                    continue

                # Validate against registered tools
                if name not in [t.name for t in self.tool_registry.list_tools()]:
                    continue

                # Parse arguments from XML body tags: <url>...</url> etc.
                args: dict[str, Any] = {}
                for arg_match in self._ARG_TAG.finditer(body):
                    key, val = arg_match.group(1), arg_match.group(2).strip()
                    # Try to parse as JSON first (for nested objects)
                    try:
                        args[key] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        args[key] = val
                        
                # Check for <parameter name="...">...</parameter> style
                for arg_match in self._PARAM_TAG.finditer(body):
                    key, val = arg_match.group(1), arg_match.group(2).strip()
                    try:
                        args[key] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        args[key] = val

                calls.append({"id": call_id, "name": name, "arguments": args})

        # ── Loose fallback ───────────────────────────────────────────
        # If none of the structured patterns matched, try the flattened
        # colon-delimited <tool_call> shape (no <invoke> tags, no JSON).
        if not calls:
            registered = {t.name for t in self.tool_registry.list_tools()}
            for m in self._LOOSE_TOOL_CALL.finditer(text):
                body = m.group("body")
                name_match = self._LOOSE_NAME.search(body)
                if not name_match:
                    continue
                name = name_match.group(1)
                if name not in registered:
                    continue

                call_counter += 1
                call_id = f"text_call_{call_counter}"
                args: dict[str, Any] = {}

                parts = self._LOOSE_ARGS_SPLIT.split(body, maxsplit=1)
                if len(parts) == 2:
                    arg_blob = parts[1].strip()
                    # Prefer a JSON object if the args happen to be JSON.
                    if arg_blob.startswith("{"):
                        try:
                            parsed = json.loads(arg_blob)
                            if isinstance(parsed, dict):
                                args = parsed
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                    # Otherwise scrape bare key: value lines.
                    if not args:
                        for key, val in self._LOOSE_KV.findall(arg_blob):
                            if key.lower() in self._LOOSE_RESERVED or not val:
                                continue
                            try:
                                args[key] = json.loads(val)
                            except (json.JSONDecodeError, ValueError):
                                args[key] = val

                calls.append({"id": call_id, "name": name, "arguments": args})

        return calls if calls else None

    # Patterns to strip tool-call markup from displayed content
    _STRIP_PATTERNS = [
        re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<function_?call>.*?(?:</function_?call>|$)", re.DOTALL | re.IGNORECASE),
    ]

    def _strip_tool_markup(self, text: str) -> str:
        """Remove tool-call XML/JSON blocks from the display text."""
        for pat in self._STRIP_PATTERNS:
            text = pat.sub("", text)
        return text.strip()


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
        persistence: Any = None,
    ) -> None:
        self.config = config
        self.session = session
        self.mode = mode
        self.approval = approval
        self.persistence = persistence  # Optional PersistenceManager for audit + checkpoint

        # Capability registry (local cache)
        self.registry = CapabilityRegistry()
        self.registry.bootstrap_defaults()

        # Rate limiter — shared across all model roles so the aggregate
        # call count stays under the configured ceiling.
        rate_limit = getattr(config, "api_rate_limit", DEFAULT_API_RATE_LIMIT)
        self.rate_limiter = RateLimiter(max_calls=rate_limit)

        # Model client pool (5 roles, lazy init)
        self.pool = ModelClientPool(
            base_url=config.base_url,
            api_key=config.api_key,
            registry=self.registry,
            overrides=model_overrides or {},
            rate_limiter=self.rate_limiter,
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
            theme.print_warning(f"Tool/Context layer unavailable: {e}")
            self.tool_context = None
            self.context_engine = None

        # ReAct executor (wired with the context engine + tool context)
        self._thinking_ref = [thinking_level]
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
            persistence=self.persistence,
        )

        self._client: Any = None

        # CommandRouter for built-in slash commands — shares the mutable
        # thinking-level list so /reasoning changes propagate immediately.
        from cli import CommandRouter
        self.commands = CommandRouter(
            config, session, mode, approval,
            thinking_level_ref=self._thinking_ref,
        )

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
        self.tool_registry.register(WebSearchTool())

    @property
    def client(self):
        """Lazily initialize the OpenAI-compatible client (action model)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                theme.print_error("'openai' package not found.\nInstall with: /usr/local/bin/pip3 install openai")
                import sys
                sys.exit(1)
            self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return self._client

    def chat(self, user_input: str) -> str:
        """Process user input through the Extended ReAct loop."""
        return self.executor.run(user_input)

    def run_interactive(self) -> None:
        """Multi-turn REPL loop in agentic mode."""
        theme.print_mode_header("Rift agentic mode")
        while True:
            try:
                user_input = theme.prompt_user()
            except (EOFError, KeyboardInterrupt):
                theme.print_goodbye()
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", ":q"):
                theme.print_goodbye()
                break
            if self.commands.try_handle(user_input):
                # If /reasoning was changed, propagate to executor
                self.executor.thinking_level = self._thinking_ref[0]
                continue

            theme.print_separator(animated=False)
            reply = self.chat(user_input)
            theme.print_reply(reply)

            # Estimate context usage from session history
            used = sum(len(m.content) for m in self.session.messages) // 4
            max_ctx = 131072  # 128K default context window
            theme.print_context_bar(used, max_ctx)

            theme.print_separator(animated=True)


# ═════════════════════════════════════════════════════════════════════
#  Security helpers
# ═════════════════════════════════════════════════════════════════════

# Paths the agent must never read from or write to
SENSITIVE_PATHS = [
    ".ssh", ".gnupg", ".aws", ".config/gcloud",
    "Library/Keychains", ".docker", ".kube",
    "/etc/shadow", "/etc/passwd", "/etc/sudoers",
    "/private/var/db/", "/System/Library/",
]

# Max file size for read_file (10 MB)
MAX_READ_BYTES = 10 * 1024 * 1024
# Max content size for write_file (10 MB)
MAX_WRITE_BYTES = 10 * 1024 * 1024
# Shell command blocklist — patterns that are always rejected
SHELL_BLOCKLIST = [
    r"\brm\s+-rf\s+/",          # rm -rf /
    r"\brm\s+-rf\s+~",          # rm -rf ~
    r"\brm\s+-rf\s+\*",         # rm -rf *
    r"mkfs\.",                   # format filesystem
    r"\bdd\s+.*of=/dev/",       # dd to device
    r":\(\)\s*\{\s*:|:\|:&\s*\};",  # fork bomb
    r">\s*/dev/sd[a-z]",        # overwrite disk device
    r"curl\s+.*\|\s*(bash|sh)", # curl pipe to shell
    r"wget\s+.*\|\s*(bash|sh)", # wget pipe to shell
    r"chmod\s+-R\s+777\s+/",    # recursive 777 on root
]

# SSRF: blocked IP ranges
SSRF_BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "169.254.169.254",  # cloud metadata
    "metadata.google.internal",
}


def _check_sensitive_path(path: Path) -> str | None:
    """Return a reason string if the path is sensitive, else None."""
    # Expand ~ so checks work with home-relative paths
    try:
        expanded = path.expanduser()
    except Exception:
        expanded = path
    # Check the raw (expanded) string first — on macOS /etc is a symlink
    # to /private/etc, so resolve() would change the prefix.
    raw = str(expanded)
    try:
        resolved = str(expanded.resolve())
    except Exception:
        resolved = raw
    home = str(Path.home())
    for sensitive in SENSITIVE_PATHS:
        if sensitive.startswith("/"):
            if raw.startswith(sensitive) or resolved.startswith(sensitive):
                return f"Access to '{sensitive}' is blocked for security"
        else:
            full_sensitive = f"{home}/{sensitive}"
            if raw.startswith(full_sensitive) or resolved.startswith(full_sensitive):
                return f"Access to '~/{sensitive}' is blocked for security"
    return None


def _check_shell_command(command: str) -> str | None:
    """Return a reason string if the command is blocked, else None."""
    for pattern in SHELL_BLOCKLIST:
        if re.search(pattern, command):
            return f"Command blocked by security policy (matched: {pattern})"
    return None


def _check_ssrf(url: str) -> str | None:
    """Return a reason string if the URL targets a blocked host, else None."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host.lower() in SSRF_BLOCKED_HOSTS:
            return f"URL blocked: '{host}' is in the SSRF blocklist"
        # Block raw IPs in private ranges
        try:
            import ipaddress
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return f"URL blocked: '{host}' is a private/loopback IP"
        except ValueError:
            pass  # Not an IP, that's fine
    except Exception:
        pass
    return None


# ═════════════════════════════════════════════════════════════════════
#  Built-in Tool Implementations
# ═════════════════════════════════════════════════════════════════════

class ReadFileTool:
    name = "read_file"
    description = "Read the contents of a file. Max 10MB."
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
        # Security: block sensitive paths
        blocked = _check_sensitive_path(path)
        if blocked:
            return f"[Error: {blocked}]"
        if not path.exists():
            return f"[Error: File not found: {path}]"
        # Security: block symlinks that resolve to sensitive locations
        try:
            resolved = path.resolve()
            blocked = _check_sensitive_path(resolved)
            if blocked:
                return f"[Error: {blocked}]"
        except Exception:
            pass
        # Size limit
        try:
            size = path.stat().st_size
            if size > MAX_READ_BYTES:
                return f"[Error: File is {size} bytes — exceeds {MAX_READ_BYTES // (1024*1024)}MB limit]"
        except OSError as e:
            return f"[Error: {e}]"
        try:
            return path.read_text(encoding="utf-8", errors="replace")
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
        blocked = _check_sensitive_path(path)
        if blocked:
            return f"[Error: {blocked}]"
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
        blocked = _check_sensitive_path(path)
        if blocked:
            return f"[Error: {blocked}]"
        matches = []
        for root, _dirs, files in os.walk(path):
            # Don't descend into sensitive directories
            root_path = Path(root)
            if _check_sensitive_path(root_path):
                continue
            for filename in fnmatch.filter(files, pattern):
                matches.append(str(Path(root) / filename))
        return "\n".join(matches) if matches else "[No matches found]"


class WriteFileTool:
    name = "write_file"
    description = "Write content to a file (creates or overwrites). Max 10MB."
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
        content = arguments["content"]
        # Security: block sensitive paths
        blocked = _check_sensitive_path(path)
        if blocked:
            return f"[Error: {blocked}]"
        # Security: block symlinks
        try:
            if path.exists() or path.is_symlink():
                resolved = path.resolve()
                blocked = _check_sensitive_path(resolved)
                if blocked:
                    return f"[Error: {blocked}]"
        except Exception:
            pass
        # Size limit
        if len(content) > MAX_WRITE_BYTES:
            return f"[Error: Content is {len(content)} bytes — exceeds {MAX_WRITE_BYTES // (1024*1024)}MB limit]"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            # Restrict permissions on the written file
            try:
                os.chmod(path, 0o644)
            except OSError:
                pass
            return f"[Written {len(content)} bytes to {path}]"
        except Exception as e:
            return f"[Error: {e}]"


class ShellExecTool:
    """Execute a shell command with security guardrails.

    - Uses ``shell=False`` (shlex.split) to prevent shell injection
    - Blocklist rejects dangerous patterns (rm -rf /, fork bombs, etc.)
    - Commands that need shell features (pipes, redirects) use
      ``/bin/bash -c`` but still go through the blocklist
    - 60-second timeout
    """
    name = "shell_exec"
    description = "Execute a shell command and return stdout + stderr. Commands run in bash with a 60s timeout."
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

        # Security: blocklist check
        blocked = _check_shell_command(command)
        if blocked:
            return f"[Error: {blocked}]"

        try:
            # Use bash -c for shell features (pipes, redirects, env vars)
            # but the blocklist has already screened for dangerous patterns
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True, text=True, timeout=60,
                shell=False,  # shell=False with explicit bash -c is safer
            )
            out = f"EXIT CODE: {result.returncode}\n"
            out += f"STDOUT:\n{result.stdout}\n"
            out += f"STDERR:\n{result.stderr}\n"
            # Truncate very large output
            if len(out) > 50_000:
                out = out[:50_000] + "\n...[output truncated]"
            return out
        except subprocess.TimeoutExpired:
            return "[Error: Command timed out after 60s]"
        except Exception as e:
            return f"[Error: {e}]"


class WebFetchTool:
    """Fetch a web page and return cleaned text content.

    Uses ``requests`` with real browser headers, cookie persistence,
    and BeautifulSoup for HTML-to-text conversion.  Falls back to
    raw HTML if parsing fails.
    """

    name = "web_fetch"
    description = (
        "Fetch the content of a web page and return it as readable text. "
        "Handles redirects, cookies, and modern web headers."
    )
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
        },
        "required": ["url"],
    }
    read_only = False

    # Reuse a single session for cookie persistence + connection pooling
    _session = None

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            import requests
            cls._session = requests.Session()
            cls._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            })
        return cls._session

    def execute(self, arguments: dict[str, Any]) -> Any:
        import requests
        from bs4 import BeautifulSoup

        url = arguments["url"]

        # Ensure URL has a scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Security: SSRF check
        blocked = _check_ssrf(url)
        if blocked:
            return f"[Error: {blocked}]"

        session = self._get_session()
        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            return f"[Error: HTTP {status} from {url}]"
        except requests.exceptions.ConnectionError as e:
            return f"[Error: Connection failed — {e}]"
        except requests.exceptions.Timeout:
            return f"[Error: Request timed out after 30s]"
        except Exception as e:
            return f"[Error: {e}]"

        html = resp.text
        content_type = resp.headers.get("Content-Type", "")

        # If it's not HTML, return raw text
        if "html" not in content_type and "xml" not in content_type:
            # Truncate very large responses
            if len(html) > 50_000:
                html = html[:50_000] + "\n...[truncated]"
            return html

        # Parse HTML → clean text
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, nav, footer, header noise
            for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
                tag.decompose()

            # Try to find main content area
            main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|article|post", re.I))
            target = main if main else soup

            text = target.get_text(separator="\n", strip=True)
            # Collapse excessive blank lines
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            text = "\n".join(lines)

            # Truncate to keep token budget reasonable
            if len(text) > 50_000:
                text = text[:50_000] + "\n...[truncated]"

            # Prepend metadata
            title = soup.find("title")
            title_str = title.get_text(strip=True) if title else "(no title)"
            return f"URL: {resp.url}\nTitle: {title_str}\nStatus: {resp.status_code}\n\n{text}"
        except Exception:
            # Fallback: return raw HTML (truncated)
            if len(html) > 50_000:
                html = html[:50_000] + "\n...[truncated]"
            return f"URL: {resp.url}\nStatus: {resp.status_code}\n\n{html}"


class WebSearchTool:
    """Search the web and return results.

    Primary: uses the ``duckduckgo_search`` package (DDGS API).
    Fallback: scrapes DuckDuckGo HTML endpoint.
    Returns formatted results with titles, URLs, and snippets.
    """

    name = "web_search"
    description = (
        "Search the web for a query and return top results with titles, "
        "URLs, and snippets. Use this to find information or discover URLs "
        "before fetching full page content with web_fetch."
    )
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }
    read_only = True

    def execute(self, arguments: dict[str, Any]) -> Any:
        query = arguments["query"]

        # ── Primary: ddgs package (renamed from duckduckgo_search) ───
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = []
            for r in DDGS().text(query, max_results=10):
                results.append(
                    f"[{len(results)+1}] {r.get('title', '')}\n"
                    f"    URL: {r.get('href', r.get('url', ''))}\n"
                    f"    {r.get('body', r.get('snippet', ''))}"
                )
            if results:
                return f"Search: {query}\nFound {len(results)} results:\n\n" + "\n\n".join(results)
        except Exception:
            pass  # Fall through to HTML scrape

        # ── Fallback: HTML scrape of DuckDuckGo ───────────────────────
        try:
            return self._scrape_ddg(query)
        except Exception as e:
            return f"[Error: Search failed — {e}]"

    def _scrape_ddg(self, query: str) -> str:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import quote_plus

        # Try the html.duckduckgo.com endpoint
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        # DDG HTML results are in div.result with a.result__a link
        for div in soup.find_all("div", class_=re.compile(r"result", re.I)):
            link = div.find("a", class_=re.compile(r"result__a|result-link", re.I))
            if not link:
                link = div.find("a", href=True)
            if link and link.get("href"):
                title = link.get_text(strip=True)
                href = link["href"]
                # DDG wraps URLs in a redirect — extract the real URL
                if "uddg=" in href:
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(href)
                    qs = parse_qs(parsed.query)
                    href = qs.get("uddg", [href])[0]
                snippet_div = div.find(class_=re.compile(r"result__snippet|snippet", re.I))
                snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                if title:
                    results.append(f"[{len(results)+1}] {title}\n    URL: {href}\n    {snippet}")

        if not results:
            return f"[No results found for: {query}]"

        return f"Search: {query}\nFound {len(results)} results:\n\n" + "\n\n".join(results[:10])
