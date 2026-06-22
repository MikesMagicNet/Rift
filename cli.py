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

from agentic_layer import AgenticAgent, ThinkingLevel

# ─── Rift project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"


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
        return self._config["model"]

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
            f"NVIDIA API key not found in Keychain or environment.\n\n"
            f"  Option 1 — Store in macOS Keychain (recommended):\n"
            f"    security add-generic-password -s '{service}' -a '{account}' -w 'nvapi-xxxxxxxx'\n\n"
            f"  Option 2 — Set an environment variable:\n"
            f"    export {env_var}='nvapi-xxxxxxxx'"
        )

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


class SessionManager:
    """Manages conversation history and session persistence.

    Each session is a JSON file stored in the configured session directory.
    Supports creating, loading, listing, and saving sessions.
    """

    def __init__(self, session_dir: Path, session_name: str | None = None) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = session_name or self._default_name()
        self.messages: list[Message] = []
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
                Message(role=m["role"], content=m["content"], timestamp=m.get("timestamp", time.time()))
                for m in data.get("messages", [])
            ]

    def save(self) -> None:
        data = {
            "session_name": self.session_name,
            "created": self.messages[0].timestamp if self.messages else time.time(),
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.messages
            ],
        }
        with open(self.session_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.save()

    def to_openai_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        """Convert session history to OpenAI-compatible message list."""
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in self.messages:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    def clear(self) -> None:
        self.messages.clear()
        self.save()

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

        print(f"\n  [APPROVAL REQUIRED] {action_type.upper()}: {description}")
        try:
            response = input("  Approve? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return response in ("y", "yes")

    def set_policy(self, policy: str) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Invalid policy '{policy}'. Choose from: {self.VALID_POLICIES}")
        self.policy = policy


# ═════════════════════════════════════════════════════════════════════
#  RiftAgent — the downstream component that receives the managers
# ═════════════════════════════════════════════════════════════════════
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

    @property
    def client(self):
        """Lazily initialize the OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                print(
                    "Error: 'openai' package not found.\n"
                    "Install with: /usr/local/bin/pip3 install openai"
                )
                sys.exit(1)
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    def chat(self, user_input: str) -> str:
        """Send user input to the model and return the response."""
        self.session.add("user", user_input)

        messages = self.session.to_openai_messages(self.config.system_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            reply = f"[Error] {e}"

        self.session.add("assistant", reply)
        return reply

    def run_oneshot(self, prompt: str) -> None:
        """Single prompt → single response."""
        print(f"\n  >> {prompt}\n")
        reply = self.chat(prompt)
        print(f"  {reply}\n")

    def run_interactive(self) -> None:
        """Multi-turn REPL loop."""
        print("\n  Rift interactive mode. Type 'exit' or Ctrl+C to quit.\n")
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
#  Argument parsing & bootstrapping
# ═════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rift",
        description="Rift — LLM agent harness (NVIDIA NIM)",
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

    # 1. ConfigManager
    config = ConfigManager(Path(args.config))
    config.override(model=args.model, temperature=args.temperature)

    if args.show_config:
        print("\n  Rift Configuration:\n")
        print(config.summary())
        print()
        return None

    # 2. SessionManager
    session_dir = PROJECT_ROOT / config._config.get("session_dir", ".sessions") # Session finding
    session_name = None if args.new_session else args.session
    session = SessionManager(session_dir, session_name) 

    if args.list_sessions: # Finds a list of saved sessions.
        sessions = session.list_sessions()
        if not sessions: 
            print("\n  No saved sessions found.\n") # NONE SAVED
        else:
            print(f"\n  Saved sessions ({session_dir}):\n") # SAVED SESSIONS FOUND
            for s in sessions:
                created = datetime.fromtimestamp(s["created"]).strftime("%Y-%m-%d %H:%M")
                print(f"    {s['name']:<30} {s['messages']:>4} msgs  {created}")
            print()
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

    print(
        f"\n  Rift initialized\n"
        f"  ─────────────────────────────────────────\n"
        f"{config.summary()}\n"
        f"  Session:      {session.session_name}\n"
        f"  Mode:         {mode}\n"
        f"  Approval:     {approval.policy}\n"
        f"  ─────────────────────────────────────────\n"
    )

    return agent


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = bootstrap(args)
    if agent is None:
        return

    # Route to the appropriate mode
    if agent.mode.is_oneshot() or args.prompt:
        prompt = args.prompt or ""
        if not prompt:
            print("  Error: --prompt is required for oneshot mode.")
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
        )
        agentic.run_interactive()


if __name__ == "__main__":
    main()
