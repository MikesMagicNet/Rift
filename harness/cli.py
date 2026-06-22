#!/usr/bin/env python3
"""
Rift — CLI Entry Point

The CLI Entry Point parses arguments and bootstraps four shared managers
(ConfigManager, SessionManager, ModeManager, and ApprovalManager)
which are injected into all downstream components.

Usage:
    /usr/local/bin/python3 cli.py                          # interactive REPL
    /usr/local/bin/python3 cli.py --prompt "Hello, Rift"   # one-shot
    /usr/local/bin/python3 cli.py --mode agentic           # agentic mode
    /usr/local/bin/python3 cli.py --list-sessions          # list saved sessions
    /usr/local/bin/python3 cli.py --session mysession      # resume a session
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure the project root (parent of harness/) is in sys.path
# so that absolute imports like `from harness...` work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agentic_layer import AgenticAgent, ThinkingLevel
from harness import theme

# ─── Rift project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "json" / "config.json"


# ═════════════════════════════════════════════════════════════════════
#  ConfigManager
# ═════════════════════════════════════════════════════════════════════
class ConfigManager:
    """Loads, validates, and exposes configuration from config.json.

    Responsible for:
      - Reading the JSON config file
      - Resolving the API key from environment variables
      - Providing capability flags to downstream components
      - Runtime overrides (model, temperature, etc.)
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config_path = config_path
        self._config: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Create one based on the Rift config schema."
            )
        with open(self.config_path) as f:
            self._config = json.load(f)
        self._validate()

    def _validate(self) -> None:
        required = ("model", "base_url", "api_key_env", "capabilities")
        for key in required:
            if key not in self._config:
                raise ValueError(f"Config missing required key: '{key}'")

    @property
    def model(self) -> str:
        model = self._config["model"]
        # Guard against accidental double-prefix typos (e.g. "nnvidia/...")
        if model.startswith("nnvidia/"):
            model = "nvidia/" + model[len("nnvidia/"):]
            self._config["model"] = model
        return model

    @property
    def base_url(self) -> str:
        return self._config["base_url"]

    def _get_key_from_keychain(self) -> str | None:
        """Attempt to retrieve the API key from macOS Keychain.

        Uses the ``security find-generic-password`` CLI.  The service and
        account names default to ``"rift"`` and ``"nvidia_api_key"`` but
        can be overridden via ``keychain_service`` / ``keychain_account``
        in config.json.

        Returns the password string on success, or ``None`` if the item
        is not found or Keychain access fails for any reason.
        """
        import subprocess

        service = self._config.get("keychain_service", "rift")
        account = self._config.get("keychain_account", "nvidia_api_key")
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s", service,
                    "-a", account,
                    "-w",           # emit only the password to stdout
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Not on macOS, or Keychain hung — fall through silently.
            pass
        return None

    @property
    def api_key(self) -> str:
        # 1. Try Apple Keychain first (macOS only).
        key = self._get_key_from_keychain()
        if key:
            return key

        # 2. Fall back to environment variable.
        env_var = self._config["api_key_env"]
        key = os.environ.get(env_var, "")
        if key:
            return key

        # 3. Neither source provided a key — raise with helpful guidance.
        service = self._config.get("keychain_service", "rift")
        account = self._config.get("keychain_account", "nvidia_api_key")
        raise EnvironmentError(
            f"API key not found in Keychain or environment.\n\n"
            f"  Option 1 — Store in macOS Keychain (recommended):\n"
            f"    security add-generic-password -s '{service}' -a '{account}' -w '<api-key>'\n\n"
            f"  Option 2 — Set an environment variable:\n"
            f"    export {env_var}='<api-key>'"
        )

    def update_keychain(self, new_key: str) -> bool:
        """Update or add the API key in the macOS Keychain."""
        import subprocess
        service = self._config.get("keychain_service", "rift")
        account = self._config.get("keychain_account", "nvidia_api_key")
        try:
            result = subprocess.run(
                ["security", "add-generic-password", "-U", "-s", service, "-a", account, "-w", new_key],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def remove_keychain(self) -> bool:
        """Remove the API key from the macOS Keychain."""
        import subprocess
        service = self._config.get("keychain_service", "rift")
        account = self._config.get("keychain_account", "nvidia_api_key")
        try:
            result = subprocess.run(
                ["security", "delete-generic-password", "-s", service, "-a", account],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @property
    def capabilities(self) -> dict[str, bool]:
        return self._config.get("capabilities", {})

    def has_capability(self, name: str) -> bool:
        return self.capabilities.get(name, False)

    @property
    def system_prompt(self) -> str:
        prompt_path = self._config.get("system_prompt_path", "")
        if not prompt_path:
            return ""
        full = PROJECT_ROOT / prompt_path
        if full.exists():
            return full.read_text(encoding="utf-8")
        return ""

    @property
    def max_tokens(self) -> int:
        return self._config.get("max_tokens", 4096)

    @property
    def temperature(self) -> float:
        return self._config.get("temperature", 0.7)

    @property
    def approval_config(self) -> dict[str, Any]:
        return self._config.get("approval", {})

    @property
    def api_rate_limit(self) -> int:
        """Max API calls per 60-second sliding window. Default: 38."""
        return self._config.get("api_rate_limit", 38)

    def override(self, **kwargs: Any) -> None:
        """Apply runtime overrides (e.g., model, temperature)."""
        for key, value in kwargs.items():
            if value is not None:
                self._config[key] = value

    def reload(self) -> None:
        self.load()

    def summary(self) -> str:
        caps = ", ".join(
            f"{k}={'on' if v else 'off'}" for k, v in self.capabilities.items()
        )
        return (
            f"  Model:        {self.model}\n"
            f"  Base URL:     {self.base_url}\n"
            f"  Max tokens:   {self.max_tokens}\n"
            f"  Temperature:  {self.temperature}\n"
            f"  Rate limit:   {self.api_rate_limit} calls/min\n"
            f"  Capabilities: {caps}"
        )


# ═════════════════════════════════════════════════════════════════════
#  SessionManager
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: list | None = None       # For assistant messages with tool calls
    tool_call_id: str | None = None      # For tool role messages


class SessionManager:
    """Manages conversation history and session persistence.

    Each session is a JSON file stored in the configured session directory.
    Supports creating, loading, listing, and saving sessions.

    Saves are debounced — ``add()`` marks the session dirty and a background
    flush writes to disk at most once per ``FLUSH_INTERVAL`` seconds (or on
    ``flush()`` / ``save()``).  Writes are atomic (temp file + rename) with
    restricted file permissions (0600).
    """

    FLUSH_INTERVAL = 2.0  # seconds — debounce window

    def __init__(self, session_dir: Path, session_name: str | None = None) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = session_name or self._default_name()
        self.messages: list[Message] = []
        self._dirty = False
        self._last_flush = 0.0
        self._load()

    @staticmethod
    def _default_name() -> str:
        return datetime.now().strftime("session_%Y%m%d_%H%M%S")

    @property
    def session_path(self) -> Path:
        return self.session_dir / f"{self.session_name}.json"

    def _load(self) -> None:
        if self.session_path.exists():
            with open(self.session_path) as f:
                data = json.load(f)
            self.messages = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m.get("timestamp", time.time()),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                for m in data.get("messages", [])
            ]

    def save(self) -> None:
        """Force an immediate flush to disk (atomic write)."""
        self._flush()

    def _maybe_flush(self) -> None:
        """Flush if dirty and the debounce window has elapsed."""
        if self._dirty and (time.time() - self._last_flush) >= self.FLUSH_INTERVAL:
            self._flush()

    def _flush(self) -> None:
        """Write session to a temp file then atomically rename.  Sets 0600 perms."""
        data = {
            "session_name": self.session_name,
            "created": self.messages[0].timestamp if self.messages else time.time(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    **({"tool_calls": m.tool_calls} if m.tool_calls else {}),
                    **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
                }
                for m in self.messages
            ],
        }
        tmp_path = self.session_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        # Restrict permissions: owner read/write only
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        # Atomic rename
        os.replace(tmp_path, self.session_path)
        self._dirty = False
        self._last_flush = time.time()

    def add(self, role: str, content: str, tool_calls: list | None = None, tool_call_id: str | None = None) -> None:
        self.messages.append(Message(role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id))
        self._dirty = True
        self._maybe_flush()

    def to_openai_messages(self, system_prompt: str = "") -> list[dict[str, Any]]:
        """Convert session history to OpenAI-compatible message list.

        Ensures any system messages stored mid-conversation are bubbled
        to the front (merged) so providers that reject non-initial system
        messages don't error out.
        """
        # Gather all system content (config + any stored system messages)
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)

        non_system: list[dict[str, Any]] = []
        for m in self.messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                msg: dict[str, Any] = {"role": m.role, "content": m.content}
                if m.tool_calls:
                    msg["tool_calls"] = m.tool_calls
                if m.tool_call_id:
                    msg["tool_call_id"] = m.tool_call_id
                non_system.append(msg)

        result: list[dict[str, Any]] = []
        if system_parts:
            result.append({"role": "system", "content": "\n\n".join(system_parts)})
        result.extend(non_system)
        return result

    def clear(self) -> None:
        self.messages.clear()
        self._dirty = True
        self._flush()

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for path in sorted(self.session_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                msg_count = len(data.get("messages", []))
                sessions.append({
                    "name": path.stem,
                    "messages": msg_count,
                    "created": data.get("created", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions


# ═════════════════════════════════════════════════════════════════════
#  ModeManager
# ═════════════════════════════════════════════════════════════════════
class ModeManager:
    """Controls operational mode — how Rift processes user input.

    Modes:
      - interactive:  Multi-turn REPL loop (default)
      - oneshot:      Single prompt → single response, then exit
      - agentic:      Autonomous mode with tool-use and approval gating
    """

    VALID_MODES = ("interactive", "oneshot", "agentic")

    def __init__(self, mode: str = "interactive") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {self.VALID_MODES}")
        self.mode = mode

    def is_interactive(self) -> bool:
        return self.mode == "interactive"

    def is_oneshot(self) -> bool:
        return self.mode == "oneshot"

    def is_agentic(self) -> bool:
        return self.mode == "agentic"

    def set_mode(self, mode: str) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {self.VALID_MODES}")
        self.mode = mode

    def __str__(self) -> str:
        return self.mode


# ═════════════════════════════════════════════════════════════════════
#  ApprovalManager
# ═════════════════════════════════════════════════════════════════════
class ApprovalManager:
    """Gates potentially-unsafe operations behind user approval.

    Policies:
      - always:      Ask before every action
      - on_request:  Ask only for actions flagged as requiring approval
      - never:       Auto-approve everything (dangerous — agentic only)

    Per-category overrides (shell, web, file) can tighten or loosen
    the policy for specific tool types.
    """

    VALID_POLICIES = ("always", "on_request", "never")

    def __init__(
        self,
        policy: str = "on_request",
        config: dict[str, Any] | None = None,
    ) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Invalid policy '{policy}'. Choose from: {self.VALID_POLICIES}")
        self.policy = policy
        config = config or {}
        self.auto_approve_safe: bool = config.get("auto_approve_safe", True)
        self.require_for_shell: bool = config.get("require_for_shell", True)
        self.require_for_web: bool = config.get("require_for_web", False)

    def needs_approval(self, action_type: str, description: str = "") -> bool:
        """Determine whether a given action requires user approval."""
        if self.policy == "never":
            return False
        if self.policy == "always":
            return True
        # on_request
        if action_type == "shell" and self.require_for_shell:
            return True
        if action_type == "web" and self.require_for_web:
            return True
        if action_type == "file" and not self.auto_approve_safe:
            return True
        return False

    def request(self, action_type: str, description: str) -> bool:
        """Prompt the user for approval. Returns True if approved."""
        if not self.needs_approval(action_type, description):
            return True

        response = theme.prompt_approval(action_type, description)
        if not response:
            return False
        return response in ("y", "yes")

    def set_policy(self, policy: str) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Invalid policy '{policy}'. Choose from: {self.VALID_POLICIES}")
        self.policy = policy


# ═════════════════════════════════════════════════════════════════════
#  CommandRouter — built-in slash commands
# ═════════════════════════════════════════════════════════════════════
class CommandRouter:
    """Intercepts ``/``-prefixed input and dispatches to built-in handlers.

    Holds live references to the four managers so changes propagate
    immediately.  Returns ``True`` if the input was consumed, ``False``
    if it should be forwarded to the model.
    """

    # ── class-level metadata (command → short description) ───────────
    COMMANDS: dict[str, str] = {
        "/help":        "Show available commands",
        "/model":       "View or change the active model",
        "/reasoning":   "Adjust thinking/reasoning depth (off, low, medium, high)",
        "/temperature": "View or set sampling temperature",
        "/mode":        "View or switch operational mode",
        "/session":     "Session management (list, new, load, save)",
        "/clear":       "Clear the current conversation history",
        "/config":      "Show the running configuration",
        "/approval":    "View or change the approval policy",
        "/tokens":      "View or set max output tokens",
        "/apikey":      "Update or remove the API key in macOS Keychain",
    }

    # Sub-options for each command (None = freeform / no sub-completions)
    COMMAND_COMPLETIONS: dict[str, list[str] | None] = {
        "/help":        None,
        "/model":       None,
        "/reasoning":   ["off", "low", "medium", "high"],
        "/temperature": None,
        "/mode":        ["interactive", "oneshot", "agentic"],
        "/session":     ["list", "new", "save", "load"],
        "/clear":       None,
        "/config":      None,
        "/approval":    ["always", "on_request", "never"],
        "/tokens":      None,
        "/apikey":      ["set", "remove"],
    }

    def __init__(
        self,
        config: "ConfigManager",
        session: "SessionManager",
        mode: "ModeManager",
        approval: "ApprovalManager",
        *,
        thinking_level_ref: list | None = None,
    ) -> None:
        self.config = config
        self.session = session
        self.mode = mode
        self.approval = approval
        # Mutable container so agentic callers can share a ThinkingLevel ref.
        # Element 0 is always the current ThinkingLevel enum value.
        from harness.agentic_layer import ThinkingLevel
        self._thinking: list = thinking_level_ref or [ThinkingLevel.MEDIUM]

        # Register hints with the theme smart-prompt system
        theme.register_command_hints(self.COMMANDS, self.COMMAND_COMPLETIONS)

    # ── public entry point ──────────────────────────────────────────
    def try_handle(self, raw_input: str) -> bool:
        """If *raw_input* is a slash command, execute it and return True."""
        if not raw_input.startswith("/"):
            return False

        # Bare "/" → show help overview
        if raw_input.strip() == "/":
            self._cmd_help("")
            return True

        parts = raw_input.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        handler = {
            "/help":        self._cmd_help,
            "/model":       self._cmd_model,
            "/reasoning":   self._cmd_reasoning,
            "/temperature": self._cmd_temperature,
            "/mode":        self._cmd_mode,
            "/session":     self._cmd_session,
            "/clear":       self._cmd_clear,
            "/config":      self._cmd_config,
            "/approval":    self._cmd_approval,
            "/tokens":      self._cmd_tokens,
            "/apikey":      self._cmd_apikey,
        }.get(cmd)

        if handler is None:
            theme.print_error(f"Unknown command: {cmd}  — type /help for a list")
            return True

        handler(arg)
        return True

    # ── individual handlers ─────────────────────────────────────────

    def _cmd_help(self, _arg: str) -> None:
        theme.print_command_table(
            "available commands",
            list(self.COMMANDS.items()),
        )

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            theme.print_command_result([("model", self.config.model)])
            theme.print_info("  usage: /model <model-name>")
            return
        old = self.config.model
        self.config.override(model=arg)
        # Force the client to be recreated on next call so the new model
        # is actually used.
        theme.print_success(f"Model changed: {old} → {arg}")

    def _cmd_reasoning(self, arg: str) -> None:
        from harness.agentic_layer import ThinkingLevel
        current = self._thinking[0]
        if not arg:
            levels = " · ".join(
                f"[{lv.value}]" if lv == current else lv.value
                for lv in ThinkingLevel
            )
            theme.print_command_result([
                ("reasoning", current.value),
                ("options", levels),
            ])
            return
        try:
            new = ThinkingLevel(arg.lower())
        except ValueError:
            valid = ", ".join(lv.value for lv in ThinkingLevel)
            theme.print_error(f"Invalid level '{arg}'. Choose from: {valid}")
            return
        old = current
        self._thinking[0] = new
        theme.print_success(f"Reasoning: {old.value} → {new.value}")

    def _cmd_temperature(self, arg: str) -> None:
        if not arg:
            theme.print_command_result([
                ("temperature", str(self.config.temperature)),
                ("range", "0.0 – 2.0"),
            ])
            theme.print_info("  usage: /temperature <value>")
            return
        try:
            t = float(arg)
            if not (0.0 <= t <= 2.0):
                raise ValueError
        except ValueError:
            theme.print_error("Temperature must be a number between 0.0 and 2.0")
            return
        old = self.config.temperature
        self.config.override(temperature=t)
        theme.print_success(f"Temperature: {old} → {t}")

    def _cmd_mode(self, arg: str) -> None:
        if not arg:
            theme.print_command_result([
                ("mode", self.mode.mode),
                ("options", " · ".join(ModeManager.VALID_MODES)),
            ])
            return
        try:
            self.mode.set_mode(arg.lower())
            theme.print_success(f"Mode set to: {self.mode.mode}")
            theme.print_warning("Mode change takes effect on next launch.")
        except ValueError:
            theme.print_error(f"Invalid mode '{arg}'. Choose from: {', '.join(ModeManager.VALID_MODES)}")

    def _cmd_session(self, arg: str) -> None:
        sub = arg.lower().split(None, 1) if arg else []
        action = sub[0] if sub else ""
        name = sub[1] if len(sub) > 1 else ""

        if action == "list":
            sessions = self.session.list_sessions()
            theme.print_sessions(sessions, str(self.session.session_dir))
        elif action == "new":
            self.session.messages.clear()
            old_name = self.session.session_name
            self.session.session_name = name or self.session._default_name()
            theme.print_success(f"New session: {self.session.session_name}")
        elif action == "save":
            self.session.save()
            theme.print_success(f"Session saved: {self.session.session_name}")
        elif action == "load" and name:
            candidate = self.session.session_dir / f"{name}.json"
            if candidate.exists():
                self.session.session_name = name
                self.session._load()
                theme.print_success(f"Loaded session: {name} ({len(self.session.messages)} messages)")
            else:
                theme.print_error(f"Session not found: {name}")
        else:
            theme.print_command_result([
                ("session", self.session.session_name),
                ("messages", str(len(self.session.messages))),
            ])
            theme.print_info("  /session list        — list saved sessions")
            theme.print_info("  /session new [name]  — start a new session")
            theme.print_info("  /session save        — save current session")
            theme.print_info("  /session load <name> — load a saved session")

    def _cmd_clear(self, _arg: str) -> None:
        count = len(self.session.messages)
        self.session.clear()
        theme.print_success(f"Cleared {count} messages from session")

    def _cmd_config(self, _arg: str) -> None:
        theme.print_command_result([
            ("model",       self.config.model),
            ("endpoint",    self.config.base_url),
            ("max tokens",  str(self.config.max_tokens)),
            ("temperature", str(self.config.temperature)),
            ("rate limit",  f"{self.config.api_rate_limit} calls/min"),
            ("reasoning",   self._thinking[0].value),
            ("approval",    self.approval.policy),
            ("session",     self.session.session_name),
            ("mode",        self.mode.mode),
        ])

    def _cmd_approval(self, arg: str) -> None:
        if not arg:
            theme.print_command_result([
                ("policy", self.approval.policy),
                ("options", " · ".join(ApprovalManager.VALID_POLICIES)),
            ])
            return
        try:
            self.approval.set_policy(arg.lower())
            theme.print_success(f"Approval policy set to: {self.approval.policy}")
        except ValueError:
            theme.print_error(f"Invalid policy '{arg}'. Choose from: {', '.join(ApprovalManager.VALID_POLICIES)}")

    def _cmd_tokens(self, arg: str) -> None:
        if not arg:
            theme.print_command_result([("max tokens", str(self.config.max_tokens))])
            theme.print_info("  usage: /tokens <number>")
            return
        try:
            n = int(arg)
            if n < 1:
                raise ValueError
        except ValueError:
            theme.print_error("Max tokens must be a positive integer")
            return
        old = self.config.max_tokens
        self.config.override(max_tokens=n)
        theme.print_success(f"Max tokens: {old} → {n}")

    def _cmd_apikey(self, arg: str) -> None:
        sub = arg.split(None, 1) if arg else []
        action = sub[0].lower() if sub else ""
        key_val = sub[1] if len(sub) > 1 else ""

        if action == "remove":
            if self.config.remove_keychain():
                theme.print_success("API key removed from macOS Keychain")
            else:
                theme.print_error("Failed to remove API key (it may not exist in Keychain)")
        elif action == "set":
            if not key_val:
                theme.print_error("Please provide the new API key: /apikey set <key>")
            else:
                if self.config.update_keychain(key_val):
                    theme.print_success("API key securely stored in macOS Keychain")
                else:
                    theme.print_error("Failed to store API key in Keychain")
        else:
            theme.print_info("  usage: /apikey set <key>   — Store a new API key")
            theme.print_info("         /apikey remove      — Remove the key from Keychain")
class RiftAgent:
    """Minimal agent that uses the four injected managers to drive chat.

    This is intentionally lightweight — the real agentic logic (tool
    execution, multi-step reasoning, etc.) will live in agentic_layer.py.
    RiftAgent here proves the injection wiring works end-to-end.
    """

    def __init__(
        self,
        config: ConfigManager,
        session: SessionManager,
        mode: ModeManager,
        approval: ApprovalManager,
    ) -> None:
        self.config = config
        self.session = session
        self.mode = mode
        self.approval = approval
        self._client = None
        self.commands = CommandRouter(config, session, mode, approval)

    @property
    def client(self):
        """Lazily initialize the OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                theme.print_error(
                    "'openai' package not found.\n"
                    "  Install with: /usr/local/bin/pip3 install openai"
                )
                sys.exit(1)
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    def chat(self, user_input: str) -> tuple[str, int, int]:
        """Send user input to the model and return (reply, used_tokens, max_ctx).

        ``used_tokens`` and ``max_ctx`` come from the API usage field when
        available, otherwise a character-based estimate is used.
        """
        self.session.add("user", user_input)

        messages = self.session.to_openai_messages(self.config.system_prompt)

        try:
            with theme.spinner("Thinking"):
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            reply = response.choices[0].message.content or ""

            # Extract token usage from API response
            usage = getattr(response, "usage", None)
            if usage is not None:
                used = getattr(usage, "total_tokens", 0) or 0
            else:
                # Rough estimate: ~4 chars per token
                used = sum(len(m.content) for m in self.session.messages) // 4
            max_ctx = 131072  # 128K default context window
        except Exception as e:
            reply = f"[Error] {e}"
            used = sum(len(m.content) for m in self.session.messages) // 4
            max_ctx = 131072

        self.session.add("assistant", reply)
        return reply, used, max_ctx

    def run_oneshot(self, prompt: str) -> None:
        """Single prompt → single response."""
        theme.print_info(f">> {prompt}")
        reply, used, max_ctx = self.chat(prompt)
        theme.print_reply(reply)
        theme.print_context_bar(used, max_ctx)

    def run_interactive(self) -> None:
        """Multi-turn REPL loop."""
        theme.print_mode_header("Rift interactive mode")
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
                continue

            theme.print_separator(animated=False)
            reply, used, max_ctx = self.chat(user_input)
            theme.print_reply(reply)
            theme.print_context_bar(used, max_ctx)
            theme.print_separator(animated=True)


# ═════════════════════════════════════════════════════════════════════
#  Argument parsing & bootstrapping
# ═════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rift",
        description="Rift — Agent Harness (NVIDIA NIM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", "-p",
        help="One-shot prompt (skips interactive REPL)",
    )
    parser.add_argument(
        "--mode", "-m",
        default="interactive",
        choices=ModeManager.VALID_MODES,
        help="Operational mode (default: interactive)",
    )
    parser.add_argument(
        "--config", "-c",
        default=str(DEFAULT_CONFIG),
        help="Path to config.json (default: ./config.json)",
    )
    parser.add_argument(
        "--session", "-s",
        help="Session name to load or create",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Force-create a new session",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List saved sessions and exit",
    )
    parser.add_argument(
        "--model",
        help="Override the model from config",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        help="Override the temperature from config",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve all actions (sets approval policy to 'never')",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print current config and exit",
    )
    return parser


def bootstrap(args: argparse.Namespace):
    """Create the four managers and inject them into RiftAgent."""

    # 0. Persistence Manager (bottom of the dependency chain)
    from persistence_layer import PersistenceManager, inject_persistence
    pm = PersistenceManager(PROJECT_ROOT)

    # 1. ConfigManager
    config = ConfigManager(Path(args.config))
    config.override(model=args.model, temperature=args.temperature)

    if args.show_config:
        theme.print_config_standalone(config.summary())
        return None

    # 2. SessionManager
    session_dir = PROJECT_ROOT / config._config.get("session_dir", ".sessions") # Session finding
    session_name = None if args.new_session else args.session
    session = SessionManager(session_dir, session_name) 

    if args.list_sessions: # Finds a list of saved sessions.
        sessions = session.list_sessions()
        theme.print_sessions(sessions, str(session_dir))
        return None

    # 3. ModeManager
    mode = ModeManager(args.mode)

    # 4. ApprovalManager
    approval_policy = "never" if args.auto_approve else None # If auto-approved it never asks for permission.
    approval = ApprovalManager(
        policy=approval_policy or config.approval_config.get("default_policy", "on_request"),
        config=config.approval_config,
    ) # If auto-approved it uses NEVER policy else it goes with config.approval_config.get("default_policy", "on_request")

    # Inject into the agent
    agent = RiftAgent(config, session, mode, approval)

    # Wire persistence layer into the agent (audit logging + checkpoints)
    inject_persistence(agent, pm)

    theme.print_banner()
    theme.print_config(
        model=config.model,
        base_url=config.base_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        capabilities=config.capabilities,
        session_name=session.session_name,
        mode=str(mode),
        approval_policy=approval.policy,
        rate_limit=config.api_rate_limit,
    )

    return (agent, pm)  # Return tuple so main() can pass persistence through


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = bootstrap(args)
    if result is None:
        return
    agent, pm = result  # Unpack the tuple

    # Route to the appropriate mode
    if agent.mode.is_oneshot() or args.prompt:
        prompt = args.prompt or ""
        if not prompt:
            theme.print_error("--prompt is required for oneshot mode.")
            sys.exit(1)
        agent.run_oneshot(prompt)
    elif agent.mode.is_interactive():
        agent.run_interactive()
    elif agent.mode.is_agentic():
        # Agentic mode: construct AgenticAgent with the same four managers
        agentic = AgenticAgent(
            config=agent.config,
            session=agent.session,
            mode=agent.mode,
            approval=agent.approval,
            thinking_level=ThinkingLevel.MEDIUM,
            persistence=pm,  # <-- wire persistence into AgenticAgent
        )
        agentic.run_interactive()

    # Flush audit log on clean exit
    if pm is not None:
        pm.close()


if __name__ == "__main__":
    main()
