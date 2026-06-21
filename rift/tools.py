"""
tools.py
========
Tool registry and execution engine for Rift.

Provides file access, shell execution, and other active-control capabilities.
All features are gated by configuration and safety rules:
  - file_access is gated by allowed_base_dirs and deny_patterns.
  - active_control is gated by explicit enable flags.
"""

from __future__ import annotations

from dependencies import (
    os,
    sys,
    subprocess,
    json,
    re,
    Any,
    Optional,
    Path,
    BASE_DIR,
    log,
)
from typing import Callable, Dict, List


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def _normalize_allowed_dirs(dirs_str: str) -> List[Path]:
    roots: List[Path] = []
    for entry in (d.strip() for d in dirs_str.split(";")):
        if not entry:
            continue
        expanded = Path(entry).expanduser()
        if not expanded.is_absolute():
            expanded = BASE_DIR / expanded
        roots.append(expanded.resolve())
    if not roots:
        roots.append(BASE_DIR.resolve())
    return roots


def _is_path_allowed(
    target: Path,
    allowed_dirs: List[Path],
    deny_patterns: List[str],
) -> bool:
    """Return True if target lives inside allowed_dirs and does not match any deny pattern.
    """
    try:
        resolved = target.resolve()
    except Exception:
        return False

    # Must be inside one of the allowed directories
    if not any(
        resolved == d or str(d) in str(resolved) for d in allowed_dirs
    ):
        return False

    # Must not match any deny pattern
    for pattern in deny_patterns:
        if resolved.match(pattern):
            return False
    return True


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry of optional tools that the Rift agent can invoke."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capabilities = config.get("capabilities", {})

    # ── helpers ──────────────────────────────────────────────
    def _file_access_ok(self) -> bool:
        return self.capabilities.get("file_access", {}).get("enabled", False)

    def _active_control_ok(self) -> bool:
        return self.capabilities.get("active_control", {}).get("enabled", False)

    def _shell_ok(self) -> bool:
        ac = self.capabilities.get("active_control", {})
        return self._active_control_ok() and ac.get("shell", False)

    # ── schema builders ──────────────────────────────────────
    def get_tool_schemas(self) -> list[dict]:
        """Return OpenAI-compatible function-style tool schemas."""
        schemas: list[dict] = []
        if self._file_access_ok():
            schemas.append(self._read_file_schema())
            schemas.append(self._write_file_schema())
            schemas.append(self._list_directory_schema())
        if self._active_control_ok():
            schemas.append(self._shell_exec_schema())
            schemas.append(self._fetch_url_schema())
        return schemas

    def _read_file_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a local file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file."
                        }
                    },
                    "required": ["path"]
                }
            }
        }

    def _write_file_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write or overwrite text to a local file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to write to."
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }

    def _list_directory_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories inside a path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to list. Defaults to the workspace root."
                        }
                    }
                }
            }
        }

    def _shell_exec_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "shell_exec",
                "description": "Execute a shell command. Use with extreme caution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute."
                        }
                    },
                    "required": ["command"]
                }
            }
        }

    def _fetch_url_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch a remote URL and return its text content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch."
                        }
                    },
                    "required": ["url"]
                }
            }
        }

    # ── execution engine ─────────────────────────────────────
    def execute(self, name: str, arguments_str: str) -> str:
        """Run a tool by name with its JSON argument blob."""
        try:
            args: Dict[str, Any] = json.loads(arguments_str) if arguments_str else {}
        except Exception:
            return "Error: could not parse tool arguments JSON."

        dispatch: Dict[str, Callable[..., str]] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "shell_exec": self._shell_exec,
            "fetch_url": self._fetch_url,
        }
        handler = dispatch.get(name, None)
        if handler is None:
            return f"Error: unknown tool '{name}'."
        return handler(args)

    # ── file access tools ──────────────────────────────────
    def _resolve_path(self, raw_path: str) -> tuple[Path, bool]:
        """Return (resolved_path, is_allowed)."""
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        resolved = path.resolve()

        fa = self.capabilities.get("file_access", {})
        allowed = _normalize_allowed_dirs(";".join(fa.get("allowed_base_dirs", ["./"])))
        deny = fa.get("deny_patterns", [])
        allowed = _is_path_allowed(resolved, allowed, deny)
        return resolved, allowed

    def _read_file(self, args: Dict[str, Any]) -> str:
        raw = args.get("path", "")
        path, allowed = self._resolve_path(raw)
        if not allowed:
            return f"Error: access to '{raw}' is outside the allowed scope or denied by pattern."
        if not path.is_file():
            return f"Error: not a file: {raw}"
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Error: {exc}"

    def _write_file(self, args: Dict[str, Any]) -> str:
        raw = args.get("path", "")
        content = args.get("content", "")
        path, allowed = self._resolve_path(raw)
        if not allowed:
            return f"Error: access to '{raw}' is outside the allowed scope or denied by pattern."
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"OK: wrote {len(content)} chars to {raw}."
        except Exception as exc:
            return f"Error: {exc}"

    def _list_directory(self, args: Dict[str, Any]) -> str:
        raw = args.get("path", ".")
        path, allowed = self._resolve_path(raw)
        if not allowed:
            return f"Error: access to '{raw}' is outside the allowed scope or denied by pattern."
        if not path.is_dir():
            return f"Error: not a directory: {raw}"
        try:
            lines: list[str] = []
            for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                kind = "dir" if item.is_dir() else "file"
                extra = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
                lines.append(f"- {item.name} [{kind}]{extra}")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as exc:
            return f"Error: {exc}"

    # ── active control tools ────────────────────────────────
    def _shell_exec(self, args: Dict[str, Any]) -> str:
        if not self._shell_ok():
            return "Error: shell execution is disabled by configuration."
        command = args.get("command", "")
        if not command:
            return "Error: empty command."
        max_dur = self.capabilities.get("active_control", {}).get("max_command_duration", 30)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=max_dur,
            )
            out = result.stdout or ""
            err = result.stderr or ""
            if result.returncode != 0:
                return f"Exit code {result.returncode}:\n{err}\n{out}"
            return out
        except subprocess.TimeoutExpired:
            return f"Error: command exceeded {max_dur}s timeout."
        except Exception as exc:
            return f"Error: {exc}"

    def _fetch_url(self, args: Dict[str, Any]) -> str:
        url = args.get("url", "")
        if not url:
            return "Error: empty URL."
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=25) as resp:
                return resp.read().decode("utf-8", errors="replace")[:8000]
        except Exception as exc:
            return f"Error fetching URL: {exc}"
