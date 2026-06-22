#!/usr/bin/env python3
"""
Rift — Persistence Layer

Unified durable storage for the agent harness.  Provides five responsibilities:

1. State Persistence     — checkpoint / restore the full ReAct loop state
2. Audit Logging         — structured JSONL record of every action & outcome
3. Session Archiving     — compress and rotate aged session files
4. Config / Registry I/O — atomic load & save of config.json and model cache
5. Retention Cleanup     — prune old audit, checkpoint, and session artifacts

Audit event taxonomy:
- Lifecycle:    session_start, session_end, model_request, model_response
- Safety:       approval_request, approval_grant, approval_deny, safety_block
- Execution:    tool_call, tool_result, tool_error
- Context:      context_compaction, memory_curate, reminder_fire
- Checkpoint:   checkpoint_save, checkpoint_restore

Design goals:
- Atomic writes everywhere (temp-file + rename)
- No circular imports: this module sits at the BOTTOM of the dependency chain.
  cli.py → agentic_layer.py → tools_plus_context.py → persistence_layer.py
- Every public method is re-entrant and safe to call from signal handlers.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ═════════════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════════════

# File permissions: owner read/write only for sensitive files
_FILE_PERMS = 0o600
_DIR_PERMS = 0o700

# Retention policies
SESSION_RETENTION_DAYS = 30
CHECKPOINT_RETENTION_DAYS = 7
AUDIT_RETENTION_DAYS = 90

# Size limits
MAX_AUDIT_FILE_BYTES = 50 * 1024 * 1024  # 50 MB — roll over at this size
MAX_SNAPSHOT_MESSAGES = 10_000


# ═════════════════════════════════════════════════════════════════════
#  Atomic I/O Primitives
# ═════════════════════════════════════════════════════════════════════

def _atomic_write(path: Path, data: str | bytes, mode: str = "w") -> None:
    """Write to a temp file then atomically rename.  Sets 0600 perms."""
    tmp = path.with_suffix(f"{path.suffix}.tmp.{uuid.uuid4().hex[:8]}")
    try:
        if isinstance(data, bytes):
            tmp.write_bytes(data)
        else:
            tmp.write_text(data, encoding="utf-8")
        os.chmod(tmp, _FILE_PERMS)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _safe_json_load(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* on any error."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return default if default is not None else {}


# ═════════════════════════════════════════════════════════════════════
#  Audit Logging
# ═════════════════════════════════════════════════════════════════════

class AuditEventType(Enum):
    """Taxonomy of auditable events."""

    # Lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_RESTORE = "checkpoint_restore"

    # Model interaction
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    MODEL_ERROR = "model_error"

    # Tool execution
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Safety
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANT = "approval_grant"
    APPROVAL_DENY = "approval_deny"
    DOOM_LOOP_WARNING = "doom_loop_warning"
    DOOM_LOOP_PAUSE = "doom_loop_pause"
    SAFETY_BLOCK = "safety_block"

    # Context management
    CONTEXT_COMPACTION = "context_compaction"
    MEMORY_CURATE = "memory_curate"
    REMINDER_FIRE = "reminder_fire"

    # User / assistant
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"


@dataclass
class AuditEntry:
    """One structured audit record."""

    event_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    iteration: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "event": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "details": self.details,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "AuditEntry":
        d = json.loads(line)
        return cls(
            event_type=d["event"],
            timestamp=d["timestamp"],
            session_id=d.get("session_id", ""),
            iteration=d.get("iteration", 0),
            details=d.get("details", {}),
        )


class AuditLogger:
    """Thread-safe(ish) structured audit log backed by JSONL.

    Rotates files when they exceed MAX_AUDIT_FILE_BYTES.
    Old files are gzip-compressed after rotation.
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = self._pick_current_file()
        self._buffer: deque[str] = deque(maxlen=100)
        self._flush_interval = 5.0  # seconds
        self._last_flush = 0.0

    def _pick_current_file(self) -> Path:
        """Return the most recent audit file, or create a new one."""
        files = sorted(self.log_dir.glob("audit_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            if files[0].stat().st_size < MAX_AUDIT_FILE_BYTES:
                return files[0]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"audit_{timestamp}.jsonl"

    def _maybe_rotate(self) -> None:
        """Rotate current file if it exceeds size limit."""
        if not self._current_file.exists():
            return
        if self._current_file.stat().st_size >= MAX_AUDIT_FILE_BYTES:
            old = self._current_file
            compressed = old.with_suffix(".jsonl.gz")
            try:
                with open(old, "rb") as f_in:
                    with gzip.open(compressed, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                old.unlink()
            except OSError:
                pass
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._current_file = self.log_dir / f"audit_{timestamp}.jsonl"

    def log(self, entry: AuditEntry) -> None:
        """Append an entry to the audit log."""
        self._maybe_rotate()
        line = entry.to_json()
        self._buffer.append(line)
        if len(self._buffer) >= 10 or (time.time() - self._last_flush) >= self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        """Write buffered lines to disk."""
        if not self._buffer:
            return
        lines = "\n".join(self._buffer) + "\n"
        with open(self._current_file, "a", encoding="utf-8") as f:
            f.write(lines)
        self._buffer.clear()
        self._last_flush = time.time()

    def close(self) -> None:
        self._flush()

    def query(
        self,
        event_types: list[AuditEventType] | None = None,
        session_id: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query the audit log with optional filters."""
        results: list[AuditEntry] = []
        files = sorted(self.log_dir.glob("audit_*.jsonl"), key=lambda p: p.stat().st_mtime)
        compressed = sorted(self.log_dir.glob("audit_*.jsonl.gz"), key=lambda p: p.stat().st_mtime)

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = AuditEntry.from_json(line)
                        except json.JSONDecodeError:
                            continue
                        if event_types and AuditEventType(entry.event_type) not in event_types:
                            continue
                        if session_id and entry.session_id != session_id:
                            continue
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        results.append(entry)
                        if len(results) >= limit:
                            return results
            except OSError:
                continue

        for gz_path in compressed:
            try:
                with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = AuditEntry.from_json(line)
                        except json.JSONDecodeError:
                            continue
                        if event_types and AuditEventType(entry.event_type) not in event_types:
                            continue
                        if session_id and entry.session_id != session_id:
                            continue
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        results.append(entry)
                        if len(results) >= limit:
                            return results
            except OSError:
                continue

        return results

    def cleanup(self, retention_days: int = AUDIT_RETENTION_DAYS) -> int:
        """Delete audit files older than *retention_days*.  Returns count deleted."""
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for pattern in ["audit_*.jsonl", "audit_*.jsonl.gz"]:
            for path in self.log_dir.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        deleted += 1
                except OSError:
                    pass
        return deleted


# ═════════════════════════════════════════════════════════════════════
#  Checkpoint Manager — save / restore full ReAct loop state
# ═════════════════════════════════════════════════════════════════════

@dataclass
class Checkpoint:
    """A full snapshot of the agent at a point in time."""

    checkpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    session_id: str = ""
    iteration: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_id=d.get("checkpoint_id", uuid.uuid4().hex[:12]),
            created_at=d.get("created_at", time.time()),
            session_id=d.get("session_id", ""),
            iteration=d.get("iteration", 0),
            messages=d.get("messages", []),
            metadata=d.get("metadata", {}),
        )


class CheckpointManager:
    """Save and restore agent state checkpoints.

    Checkpoints are stored as individual JSON files named
    ``checkpoint_<id>.json``.  Recent checkpoints are kept;
    older ones are pruned automatically.
    """

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 50) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save(self, checkpoint: Checkpoint, audit: AuditLogger | None = None) -> Path:
        """Persist a checkpoint to disk.  Returns the file path."""
        path = self.checkpoint_dir / f"checkpoint_{checkpoint.checkpoint_id}.json"
        _atomic_write(path, json.dumps(checkpoint.to_dict(), indent=2))

        if audit:
            audit.log(AuditEntry(
                event_type=AuditEventType.CHECKPOINT_SAVE.value,
                session_id=checkpoint.session_id,
                iteration=checkpoint.iteration,
                details={"checkpoint_id": checkpoint.checkpoint_id, "path": str(path)},
            ))

        self._prune_old()
        return path

    def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID.  Returns None if not found."""
        path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
        if not path.exists():
            return None
        data = _safe_json_load(path)
        return Checkpoint.from_dict(data)

    def list_checkpoints(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """List available checkpoints, optionally filtered by session_id."""
        checkpoints = []
        for path in sorted(self.checkpoint_dir.glob("checkpoint_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = _safe_json_load(path)
                if session_id and data.get("session_id") != session_id:
                    continue
                checkpoints.append({
                    "checkpoint_id": data.get("checkpoint_id"),
                    "created_at": data.get("created_at"),
                    "session_id": data.get("session_id"),
                    "iteration": data.get("iteration"),
                    "path": str(path),
                })
            except Exception:
                continue
        return checkpoints

    def restore_latest(self, session_id: str | None = None) -> Checkpoint | None:
        """Load the most recent checkpoint, optionally filtered by session_id."""
        checkpoints = self.list_checkpoints(session_id=session_id)
        if not checkpoints:
            return None
        latest = checkpoints[0]
        return self.load(latest["checkpoint_id"])

    def _prune_old(self) -> None:
        """Remove oldest checkpoints beyond the retention limit."""
        all_paths = sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(all_paths) > self.max_checkpoints:
            for old in all_paths[self.max_checkpoints:]:
                try:
                    old.unlink()
                except OSError:
                    pass

    def cleanup(self, retention_days: int = CHECKPOINT_RETENTION_DAYS) -> int:
        """Delete checkpoints older than *retention_days*.  Returns count deleted."""
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for path in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                pass
        return deleted


# ═════════════════════════════════════════════════════════════════════
#  Session Archiver
# ═════════════════════════════════════════════════════════════════════

class SessionArchiver:
    """Compress and archive old session files to reclaim disk space."""

    def __init__(self, session_dir: Path, archive_dir: Path | None = None) -> None:
        self.session_dir = session_dir
        self.archive_dir = archive_dir or session_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def archive_old_sessions(self, retention_days: int = SESSION_RETENTION_DAYS) -> int:
        """Compress session files older than *retention_days* into the archive."""
        cutoff = time.time() - (retention_days * 86400)
        archived = 0
        for path in self.session_dir.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                dest = self.archive_dir / f"{path.stem}.json.gz"
                try:
                    with open(path, "rb") as f_in:
                        with gzip.open(dest, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    path.unlink()
                    archived += 1
                except OSError:
                    pass
        return archived

    def restore_session(self, name: str) -> Path | None:
        """Restore an archived session file to the live session directory."""
        archived = self.archive_dir / f"{name}.json.gz"
        if not archived.exists():
            return None
        dest = self.session_dir / f"{name}.json"
        try:
            with gzip.open(archived, "rb") as f_in:
                dest.write_bytes(f_in.read())
            return dest
        except OSError:
            return None


# ═════════════════════════════════════════════════════════════════════
#  Unified Persistence Manager
# ═════════════════════════════════════════════════════════════════════

class PersistenceManager:
    """Single entry point for ALL durable storage in Rift.

    Owned by the top-level bootstrap (cli.py) and injected into every
    component that needs durable state or audit logging.

    Usage::

        pm = PersistenceManager(project_root)
        pm.audit.log(AuditEntry(...))
        cp = pm.checkpoint.save(Checkpoint(...))
        restored = pm.checkpoint.restore_latest(session_id="abc")
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

        self.data_dir = project_root / ".data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.data_dir, _DIR_PERMS)
        except OSError:
            pass

        # Sub-managers
        self.audit = AuditLogger(self.data_dir / "audit")
        self.checkpoint = CheckpointManager(self.data_dir / "checkpoints")
        self.archiver = SessionArchiver(project_root / ".sessions")

        # Registry cache path (used by CapabilityRegistry in agentic_layer)
        self.model_cache_path = project_root / ".model_cache.json"

    # ── Convenience: config persistence ───────────────────────────────

    def load_config(self) -> dict[str, Any]:
        """Load the project's config.json if it exists."""
        config_path = self.project_root / "config.json"
        return _safe_json_load(config_path)

    def save_config(self, config: dict[str, Any]) -> None:
        """Atomically write the project's config.json."""
        config_path = self.project_root / "config.json"
        _atomic_write(config_path, json.dumps(config, indent=2))

    # ── Convenience: registry cache ──────────────────────────────────

    def load_model_cache(self) -> dict[str, Any]:
        """Load the model capability cache."""
        return _safe_json_load(self.model_cache_path)

    def save_model_cache(self, cache: dict[str, Any]) -> None:
        """Atomically write the model capability cache."""
        _atomic_write(self.model_cache_path, json.dumps(cache, indent=2))

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush all pending writes.  Call on exit."""
        self.audit.close()

    def cleanup(self) -> dict[str, int]:
        """Run retention cleanup on all subsystems.  Returns counts per subsystem."""
        return {
            "audit": self.audit.cleanup(),
            "checkpoints": self.checkpoint.cleanup(),
            "archived_sessions": self.archiver.archive_old_sessions(),
        }


# ═════════════════════════════════════════════════════════════════════
#  Integration helpers — wire PersistenceManager into existing flow
# ═════════════════════════════════════════════════════════════════════

def inject_persistence(agent: Any, pm: PersistenceManager) -> None:
    """Attach a PersistenceManager to an agent instance.

    Sets ``agent.persistence`` and monkey-patches the agent's chat
    method so that every user/assistant exchange is automatically
    audited.  Safe to call multiple times (idempotent).
    """
    if hasattr(agent, "persistence"):
        return  # Already injected

    agent.persistence = pm  # type: ignore[attr-defined]

    # Wrap the chat method to auto-log messages
    original_chat = agent.chat

    def _audited_chat(user_input: str) -> str:
        session_id = ""
        if hasattr(agent, "session") and agent.session is not None:
            session_id = getattr(agent.session, "session_name", "")
        pm.audit.log(AuditEntry(
            event_type=AuditEventType.USER_MESSAGE.value,
            session_id=session_id,
            details={"input_length": len(user_input)},
        ))
        response = original_chat(user_input)
        pm.audit.log(AuditEntry(
            event_type=AuditEventType.ASSISTANT_MESSAGE.value,
            session_id=session_id,
            details={"response_length": len(response)},
        ))
        return response

    agent.chat = _audited_chat  # type: ignore[method-assign]


def build_persistence_manager(project_root: Path | str) -> PersistenceManager:
    """Factory: create a fully initialised PersistenceManager."""
    root = Path(project_root).resolve()
    return PersistenceManager(root)