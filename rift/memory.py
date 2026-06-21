"""
memory.py
=========
Bounded local memory for Rift.

Security posture:
    - memory.md is created with user-only permissions where the OS supports it.
    - memory injection is capped by max_chars so old transcripts do not blow up
      context or leak excessive local history into every request.
    - conversation transcript saving is controlled by config.json, not automatic.
"""

from dependencies import (
    os,
    log,
    datetime,
    Optional,
    Path,
    MEMORY_FILE,
)


class Memory:
    """Read and append to memory.md with bounded context injection."""

    def __init__(self, path: Optional[Path] = None, max_chars: int = 4000):
        self.path = path or MEMORY_FILE
        self.max_chars = max(0, int(max_chars))
        self._ensure_exists()
        self._secure_permissions()

    def _ensure_exists(self) -> None:
        if not self.path.exists():
            self.path.write_text(
                f"# Rift Memory\n\n_Created {datetime.now():%Y-%m-%d %H:%M:%S}_\n\n"
            )
            log.info("Created memory file at %s", self.path)

    def _secure_permissions(self) -> None:
        """Best-effort chmod so local memory is not world-readable."""
        try:
            os.chmod(self.path, 0o600)
        except Exception as exc:
            log.debug("Could not chmod memory file: %s", exc)

    def read(self) -> str:
        content = self.path.read_text()
        log.debug("Read %d chars from memory", len(content))
        return content

    def context(self) -> str:
        """Return the last max_chars of memory for prompt context."""
        if self.max_chars <= 0:
            return ""
        content = self.read().strip()
        if len(content) <= self.max_chars:
            return content
        return content[-self.max_chars:]

    def append(self, entry: str) -> None:
        if not entry.strip():
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = f"\n## {timestamp}\n\n{entry.strip()}\n"
        with self.path.open("a") as f:
            f.write(block)
        self._secure_permissions()
        log.info("Appended %d chars to memory", len(entry))

    def write(self, content: str) -> None:
        self.path.write_text(content)
        self._secure_permissions()
        log.info("Overwrote memory with %d chars", len(content))

    def self_improve(self, suggestion: str) -> None:
        """Record a self-improvement suggestion for later review."""
        self.append(f"**SELF-IMPROVEMENT SUGGESTION:**\n{suggestion}")


def main():
    mem = Memory()
    print(mem.read())


if __name__ == "__main__":
    main()
