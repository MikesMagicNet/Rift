#!/usr/bin/env python3
"""
Rift — CLI Theme Engine

Minimal, clean terminal UI inspired by modern CLI tools.
Mid-grey background wash · muted palette · whitespace-driven layout.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator


# ─────────────────────────────────────────────────────────────────────
#  ANSI primitives
# ─────────────────────────────────────────────────────────────────────

def _esc(code: str) -> str:
    return f"\033[{code}m"

RESET      = _esc("0")
BOLD       = _esc("1")
DIM        = _esc("2")
ITALIC     = _esc("3")

# ── Foreground palette (muted, sophisticated) ────────────────────────
WHITE      = _esc("38;5;255")        # near-white for primary text
LIGHT      = _esc("38;5;252")        # slightly off-white
GREY       = _esc("38;5;245")        # mid-grey — labels, secondary
DARK_GREY  = _esc("38;5;240")        # dark grey — rules, borders
FAINT      = _esc("38;5;237")        # very faint — decorative only
BLUE       = _esc("38;5;111")        # soft blue — accent
TEAL       = _esc("38;5;80")         # teal — AI prompt, branding
GREEN      = _esc("38;5;114")        # muted green — success
RED        = _esc("38;5;174")        # muted rose — errors
AMBER      = _esc("38;5;179")        # muted gold — warnings
VIOLET     = _esc("38;5;141")        # muted violet — special

# ── Background ───────────────────────────────────────────────────────
BG_GREY    = _esc("48;5;236")        # mid-grey background wash
BG_DARK    = _esc("48;5;234")        # darker background for contrast
BG_RESET   = _esc("49")              # reset background only

# ── Semantic aliases ─────────────────────────────────────────────────
TEXT       = LIGHT                    # default body text
MUTED      = GREY                    # secondary / hint text
RULE       = DARK_GREY               # horizontal rules
ACCENT     = BLUE                    # headings, emphasis
BRAND      = TEAL                    # brand colour (prompts, logo)
SUCCESS    = GREEN
ERROR      = RED
WARN       = AMBER
HINT       = GREY
LABEL      = GREY
VALUE      = LIGHT


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)

def term_width() -> int:
    try:
        return min(shutil.get_terminal_size().columns, 100)
    except Exception:
        return 80

def _rule(char: str = "─") -> str:
    """A subtle horizontal rule."""
    w = term_width() - 4
    return f"  {FAINT}{char * w}{RESET}"

def _pad_bg(text: str, width: int | None = None) -> str:
    """Pad text to fill terminal width with background colour intact."""
    w = (width or term_width()) - 4
    visible = len(_strip_ansi(text))
    pad = max(0, w - visible)
    return text + " " * pad


# ─────────────────────────────────────────────────────────────────────
#  Banner — clean, minimal
# ─────────────────────────────────────────────────────────────────────

LOGO_LINES: list[str] = [
    " ▄▄▄  ▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄▄▄",
    " ▓▓▓▌  ▀▓▓▓  ▓▓▀   ▀▓▓▀",
    " ▓▓▀▓▓   ▓▓  ▓▓▀▀   ▓▓ ",
    " ▓▓ ▀▓▌ ▄▓▓▄ ▓▓    ▄▓▓▄",
]


def banner() -> str:
    lines: list[str] = [""]

    for art_line in LOGO_LINES:
        lines.append(f"  {BRAND}{BOLD}{art_line}{RESET}")

    lines.append("")
    lines.append(f"  {MUTED}Agent Harness · NVIDIA NIM{RESET}")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  Config block — grey-background panel, no borders
# ─────────────────────────────────────────────────────────────────────

def config_block(
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    capabilities: dict[str, bool],
    session_name: str,
    mode: str,
    approval_policy: str,
    rate_limit: int = 38,
) -> str:
    w = term_width()
    lines: list[str] = []

    def row(key: str, val: str) -> str:
        content = f"  {LABEL}{key:<14}{RESET} {VALUE}{val}{RESET}"
        return f"  {BG_GREY} {_pad_bg(content, w - 2)} {RESET}"

    def blank() -> str:
        return f"  {BG_GREY}{' ' * (w - 4)}{RESET}"

    lines.append(blank())
    lines.append(row("model", model))
    lines.append(row("endpoint", base_url))
    lines.append(row("max tokens", str(max_tokens)))
    lines.append(row("temperature", str(temperature)))
    lines.append(row("rate limit", f"{rate_limit} calls/min"))

    # Capabilities — inline dots
    caps = "  ".join(
        f"{GREEN}●{RESET}{BG_GREY} {LABEL}{k}{RESET}" if v
        else f"{DARK_GREY}○ {k}{RESET}"
        for k, v in capabilities.items()
    )
    cap_line = f"  {LABEL}{'capabilities':<14}{RESET} {caps}"
    lines.append(f"  {BG_GREY} {_pad_bg(cap_line, w - 2)} {RESET}")

    lines.append(blank())

    # Session info — slightly different shade
    lines.append(f"  {BG_DARK} {_pad_bg(f'  {MUTED}session{RESET}        {DARK_GREY}{session_name}{RESET}', w - 2)} {RESET}")
    lines.append(f"  {BG_DARK} {_pad_bg(f'  {MUTED}mode{RESET}           {DARK_GREY}{mode}{RESET}', w - 2)} {RESET}")
    lines.append(f"  {BG_DARK} {_pad_bg(f'  {MUTED}approval{RESET}       {DARK_GREY}{approval_policy}{RESET}', w - 2)} {RESET}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  Styled output
# ─────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print(banner())

def print_config(**kwargs: Any) -> None:
    print(config_block(**kwargs))
    print()

def print_mode_header(mode_label: str) -> None:
    print()
    print(_rule())
    print(f"  {MUTED}{mode_label}  ·  type {DARK_GREY}exit{MUTED} to quit{RESET}")
    print(_rule())

def print_goodbye() -> None:
    print(f"\n  {MUTED}goodbye{RESET}\n")

def print_reply(text: str) -> None:
    w = term_width() - 10
    wrapped = textwrap.fill(
        text,
        width=max(40, w),
        initial_indent="",
        subsequent_indent="       ",
    )
    print(f"\n  {BRAND}{BOLD}rift{RESET} {MUTED}›{RESET} {TEXT}{wrapped}{RESET}\n")

def print_error(text: str) -> None:
    print(f"\n  {ERROR}✕{RESET} {TEXT}{text}{RESET}\n")

def print_warning(text: str) -> None:
    print(f"  {WARN}⚠{RESET} {MUTED}{text}{RESET}")

def print_info(text: str) -> None:
    print(f"  {MUTED}{text}{RESET}")

def print_success(text: str) -> None:
    print(f"  {SUCCESS}✓{RESET} {TEXT}{text}{RESET}")

def print_command_result(rows: list[tuple[str, str]]) -> None:
    """Print a key → value block for slash-command output."""
    print()
    for key, val in rows:
        print(f"  {LABEL}{key:<16}{RESET} {VALUE}{val}{RESET}")
    print()

def print_command_table(heading: str, items: list[tuple[str, str]]) -> None:
    """Print a compact two-column listing for slash-command help."""
    print()
    print(f"  {MUTED}{heading}{RESET}")
    print(_rule())
    for cmd, desc in items:
        print(f"  {BRAND}{cmd:<18}{RESET} {MUTED}{desc}{RESET}")
    print()


# ── Separator pulse ramp (pure teal — no grey/white spots) ───────
_SEPARATOR_PULSE: list[int] = [
    237, 23, 29, 30, 36, 37, 43, 44, 80, 116, 80, 44, 43, 37, 36, 30, 29, 23, 237,
]


def print_separator(animated: bool = True) -> None:
    """Print a horizontal separator line between message blocks.

    When *animated* is True and stdout is a TTY, a teal gradient pulse
    sweeps left-to-right then settles into a static faint rule.
    """
    w = term_width() - 4
    if not animated or not sys.stdout.isatty():
        print(f"  {FAINT}{'─' * w}{RESET}")
        return

    pulse = _SEPARATOR_PULSE
    p_len = len(pulse)

    # Skip every other column for faster sweep
    for pos in range(0, w + p_len, 2):
        parts = []
        for col in range(w):
            idx = col - (pos - p_len)
            if 0 <= idx < p_len:
                parts.append(f"{_esc(f'38;5;{pulse[idx]}')}─")
            else:
                parts.append(f"{FAINT}─")
        sys.stdout.write(f"\r  " + "".join(parts) + RESET)
        sys.stdout.flush()
        time.sleep(0.002)
    sys.stdout.write(f"\r  {FAINT}{'─' * w}{RESET}\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────
#  Context usage bar
# ─────────────────────────────────────────────────────────────────────

def print_context_bar(used_tokens: int, max_tokens: int) -> None:
    """Print a sleek context-window usage bar below model replies."""
    if max_tokens <= 0:
        return
    pct = min(used_tokens / max_tokens, 1.0)
    bar_w = min(term_width() - 24, 36)
    filled = int(bar_w * pct)
    empty = bar_w - filled

    # Colour shifts with pressure: teal → amber → rose
    if pct < 0.50:
        bar_fg = 44   # teal
    elif pct < 0.80:
        bar_fg = 179  # amber
    else:
        bar_fg = 174  # rose

    bar = (
        f"{_esc(f'38;5;{bar_fg}')}"
        f"{'━' * filled}"
        f"{FAINT}{'╌' * empty}"
        f"{RESET}"
    )
    pct_s = f"{pct * 100:.0f}%"
    tok_s = f"{used_tokens:,}/{max_tokens:,}"
    print(f"  {DARK_GREY}ctx{RESET} {bar} {DARK_GREY}{pct_s}  {FAINT}{tok_s}{RESET}")


# ─────────────────────────────────────────────────────────────────────
#  Active separator — 24 / 7 loop on the newest separator line
# ─────────────────────────────────────────────────────────────────────

class _ActiveSeparator:
    """Continuously animate the most-recent separator line.

    Writes go through ``/dev/tty`` so readline / input() is undisturbed.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tty: Any = None

    def start(self) -> None:
        self.stop()  # idempotent
        if not sys.stdout.isatty():
            return
        try:
            self._tty = open("/dev/tty", "w")
        except OSError:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Freeze to a static faint rule
        if self._tty is not None:
            try:
                w = term_width() - 4
                self._tty.write(
                    "\0337"  # DEC save
                    "\033[1A"  # up 1
                    f"\r  {FAINT}{'─' * w}{RESET}"
                    "\0338"  # DEC restore
                )
                self._tty.flush()
                self._tty.close()
            except OSError:
                pass
            self._tty = None

    def _loop(self) -> None:
        pulse = _SEPARATOR_PULSE
        p_len = len(pulse)
        idx = 0
        tty = self._tty
        while not self._stop.is_set():
            w = term_width() - 4
            parts: list[str] = []
            for col in range(w):
                ci = col - (idx % (w + p_len) - p_len)
                if 0 <= ci < p_len:
                    parts.append(f"\033[38;5;{pulse[ci]}m─")
                else:
                    parts.append(f"{FAINT}─")
            try:
                tty.write(
                    "\0337\033[1A\r  "
                    + "".join(parts)
                    + RESET
                    + "\0338"
                )
                tty.flush()
            except OSError:
                break
            idx += 1
            self._stop.wait(0.04)  # ~25 fps


_active_sep = _ActiveSeparator()


def start_active_separator() -> None:
    """Begin the 24/7 teal pulse on the most-recent separator."""
    _active_sep.start()


def stop_active_separator() -> None:
    """Freeze the active separator to a static faint rule."""
    _active_sep.stop()


# ─────────────────────────────────────────────────────────────────────
#  Smart Prompt — live slash-command suggestions
# ─────────────────────────────────────────────────────────────────────

# Module-level command hint registry (populated by CommandRouter on init)
_command_hints: dict[str, str] = {}
_command_completions: dict[str, list[str] | None] = {}


def register_command_hints(
    commands: dict[str, str],
    completions: dict[str, list[str] | None],
) -> None:
    """Register slash-command metadata for the smart prompt."""
    global _command_hints, _command_completions
    _command_hints = dict(commands)
    _command_completions = dict(completions)


def _filter_suggestions(text: str) -> list[tuple[str, str]]:
    """Return matching ``(value, description)`` pairs for the current input."""
    if not text.startswith("/"):
        return []

    if " " in text:
        # Argument / sub-command completion
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg_prefix = parts[1].lower() if len(parts) > 1 else ""
        subs = _command_completions.get(cmd)
        if subs:
            return [(s, "") for s in subs if s.lower().startswith(arg_prefix)]
        return []
    else:
        # Command-name completion
        prefix = text.lower()
        return [
            (cmd, desc)
            for cmd, desc in _command_hints.items()
            if cmd.startswith(prefix)
        ]


def _read_key(fd: int) -> str:
    """Read one keypress, decoding multi-byte escape sequences."""
    import select as _sel

    ch = os.read(fd, 1)
    if ch == b"\x1b":
        if _sel.select([fd], [], [], 0.05)[0]:
            ch2 = os.read(fd, 1)
            if ch2 == b"[":
                ch3 = os.read(fd, 1)
                if ch3 == b"A":
                    return "UP"
                if ch3 == b"B":
                    return "DOWN"
                if ch3 == b"C":
                    return "RIGHT"
                if ch3 == b"D":
                    return "LEFT"
                while _sel.select([fd], [], [], 0.01)[0]:
                    os.read(fd, 1)
        return "ESC"
    return ch.decode("utf-8", errors="replace")


def _build_separator_frame(anim_idx: int) -> str:
    """Build one frame of the animated teal-pulse separator."""
    w = term_width() - 4
    pulse = _SEPARATOR_PULSE
    p_len = len(pulse)
    parts: list[str] = []
    for col in range(w):
        ci = col - (anim_idx % (w + p_len) - p_len)
        if 0 <= ci < p_len:
            parts.append(f"\033[38;5;{pulse[ci]}m\u2500")
        else:
            parts.append(f"{FAINT}\u2500")
    return "  " + "".join(parts) + RESET


def _animate_separator_only(anim_idx: int) -> None:
    """Update only the separator line above the prompt (DEC save/restore)."""
    sys.stdout.write(
        "\0337"                              # DEC save cursor
        "\033[1A\r"                          # up 1, column 0
        + _build_separator_frame(anim_idx) +
        "\0338"                              # DEC restore cursor
    )
    sys.stdout.flush()


def _render_prompt(buf: list[str], selected: int, anim_idx: int = 0) -> int:
    """Redraw separator + prompt line + suggestion panel.

    Returns the number of panel lines drawn below the prompt.
    """
    text = "".join(buf)

    # 1. Animate separator (one line above)
    sys.stdout.write("\033[1A\r")
    sys.stdout.write(_build_separator_frame(anim_idx))

    # 2. Prompt line — move down, redraw, erase rest of screen
    sys.stdout.write(f"\n  {WHITE}{BOLD}>{RESET} {text}\033[J")

    # 3. Suggestion panel
    suggestions = _filter_suggestions(text)
    panel = 0
    if suggestions:
        shown = suggestions[:8]
        for i, (cmd, desc) in enumerate(shown):
            if i == selected:
                if desc:
                    sys.stdout.write(
                        f"\n  {BRAND}\u203a {cmd:<16}{RESET} {MUTED}{desc}{RESET}"
                    )
                else:
                    sys.stdout.write(f"\n  {BRAND}\u203a {cmd}{RESET}")
            else:
                if desc:
                    sys.stdout.write(
                        f"\n    {DARK_GREY}{cmd:<16}{RESET} {FAINT}{desc}{RESET}"
                    )
                else:
                    sys.stdout.write(f"\n    {DARK_GREY}{cmd}{RESET}")
            panel += 1

        # Move cursor back up to the prompt line
        sys.stdout.write(f"\033[{panel}A")

    # 4. Position cursor at end of typed text
    col = 4 + len(text)
    sys.stdout.write(f"\r\033[{col}C")
    sys.stdout.flush()
    return panel


def _freeze_separator() -> None:
    """Replace the animated separator above with a static faint rule."""
    w = term_width() - 4
    sys.stdout.write(
        f"\033[1A\r  {FAINT}{'\u2500' * w}{RESET}"
        f"\033[1B\r"
    )


def _smart_prompt() -> str:
    """Character-by-character prompt with live suggestions and animated separator."""
    import tty
    import termios
    import select as _sel

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf: list[str] = []
    selected = 0
    anim_idx = 0

    sys.stdout.write(f"  {WHITE}{BOLD}>{RESET} ")
    sys.stdout.flush()

    try:
        tty.setcbreak(fd)

        while True:
            # Poll: 40 ms timeout → ~25 fps separator animation
            ready = _sel.select([fd], [], [], 0.04)[0]

            if not ready:
                # No input — animate separator only
                anim_idx += 1
                _animate_separator_only(anim_idx)
                continue

            # ── Input available ──────────────────────────────────────
            try:
                key = _read_key(fd)
            except KeyboardInterrupt:
                _freeze_separator()
                sys.stdout.write("\r\033[J\n")
                sys.stdout.flush()
                raise

            if key in ("\r", "\n"):
                text = "".join(buf)
                _freeze_separator()
                sys.stdout.write(f"\r  {WHITE}{BOLD}>{RESET} {text}\033[J\n")
                sys.stdout.flush()
                return text.strip()

            elif key == "\x03":
                _freeze_separator()
                sys.stdout.write("\r\033[J\n")
                sys.stdout.flush()
                raise KeyboardInterrupt

            elif key == "\x04":
                _freeze_separator()
                sys.stdout.write("\r\033[J\n")
                sys.stdout.flush()
                raise EOFError

            elif key in ("\x7f", "\x08"):
                if buf:
                    buf.pop()
                    selected = 0

            elif key == "\t":
                text = "".join(buf)
                suggestions = _filter_suggestions(text)
                if suggestions:
                    idx = min(selected, len(suggestions) - 1)
                    chosen = suggestions[idx][0]
                    if " " not in text:
                        buf = list(chosen + " ")
                    else:
                        cmd_part = text.split(None, 1)[0]
                        buf = list(cmd_part + " " + chosen)
                    selected = 0

            elif key == "UP":
                if selected > 0:
                    selected -= 1

            elif key == "DOWN":
                suggestions = _filter_suggestions("".join(buf))
                if selected < len(suggestions) - 1:
                    selected += 1

            elif key in ("ESC", "LEFT", "RIGHT"):
                continue

            elif len(key) == 1 and key.isprintable():
                buf.append(key)
                selected = 0

            else:
                continue

            anim_idx += 1
            _render_prompt(buf, selected, anim_idx)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def prompt_user() -> str:
    """Prompt for user input with live slash-command suggestions."""
    if sys.stdin.isatty() and _command_hints:
        return _smart_prompt()
    try:
        return input(f"  {WHITE}{BOLD}>{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        raise


def prompt_approval(action_type: str, description: str) -> str:
    print()
    print(f"  {WARN}{BOLD}approval required{RESET}")
    print(f"  {MUTED}{action_type}{RESET}  {DARK_GREY}{description}{RESET}")
    try:
        return input(f"  {WARN}approve? [y/N]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return ""


# ─────────────────────────────────────────────────────────────────────
#  Sessions
# ─────────────────────────────────────────────────────────────────────

def print_sessions(sessions: list[dict[str, Any]], session_dir: str) -> None:
    from datetime import datetime

    if not sessions:
        print(f"\n  {MUTED}no saved sessions{RESET}\n")
        return

    print()
    print(f"  {MUTED}sessions  ·  {DARK_GREY}{session_dir}{RESET}")
    print(_rule())

    for s in sessions:
        created = datetime.fromtimestamp(s["created"]).strftime("%Y-%m-%d %H:%M")
        name = s["name"]
        msgs = s["messages"]
        print(f"  {TEXT}{name:<30}{RESET}  {MUTED}{msgs:>4} msgs{RESET}  {DARK_GREY}{created}{RESET}")

    print()


def print_config_standalone(summary_text: str) -> None:
    print()
    print(f"  {MUTED}configuration{RESET}")
    print(_rule())
    for line in summary_text.strip().splitlines():
        stripped = line.strip()
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            print(f"  {LABEL}{key.strip():<14}{RESET} {VALUE}{val.strip()}{RESET}")
        else:
            print(f"  {TEXT}{stripped}{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────
#  Spinner — minimal, clean
# ─────────────────────────────────────────────────────────────────────

# Subtle dot-trail frames — a single dot moves across a track
_DOT_TRAIL: list[str] = [
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
]

# Colour ramps — gentle breathing between two tones
_RAMP_BLUE:   list[int] = [60, 67, 68, 75, 111, 117, 153, 117, 111, 75, 68, 67]
_RAMP_TEAL:   list[int] = [23, 30, 37, 44, 80, 116, 80, 44, 37, 30]
_RAMP_GREEN:  list[int] = [22, 28, 34, 41, 78, 114, 78, 41, 34, 28]
_RAMP_AMBER:  list[int] = [130, 136, 172, 179, 215, 179, 172, 136]
_RAMP_VIOLET: list[int] = [53, 60, 97, 134, 141, 177, 141, 134, 97, 60]
_RAMP_GREY:   list[int] = [238, 240, 242, 244, 246, 248, 246, 244, 242, 240]

def _fg256(code: int) -> str:
    return f"\033[38;5;{code}m"

# Label → (colour ramp, lowercase action text)
_SPINNER_CONFIG: dict[str, tuple[list[int], str]] = {
    "thinking":    (_RAMP_BLUE,   "thinking"),
    "planning":    (_RAMP_VIOLET, "planning"),
    "coding":      (_RAMP_GREEN,  "coding"),
    "executing":   (_RAMP_AMBER,  "executing"),
    "analyzing":   (_RAMP_TEAL,   "analyzing"),
    "summarizing": (_RAMP_GREY,   "summarizing"),
    "refining":    (_RAMP_VIOLET, "refining"),
    "critiquing":  (_RAMP_AMBER,  "critiquing"),
}

_DEFAULT_SPINNER = (_RAMP_BLUE, "working")


class Spinner:
    """Minimal threaded spinner — breathing colour, braille dot, elapsed time.

    While running, also animates a teal pulse on the separator line
    immediately above the spinner row (the most-recent separator).

    Usage::

        with Spinner("Thinking"):
            long_running_call()
    """

    FPS = 16

    def __init__(self, label: str = "Working", style: str | None = None) -> None:
        self.label = label
        key = (style or label).lower()
        self.ramp, self.action = _SPINNER_CONFIG.get(key, _DEFAULT_SPINNER)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def _render_frame(self, idx: int, elapsed: float) -> str:
        colour = _fg256(self.ramp[idx % len(self.ramp)])
        dot = _DOT_TRAIL[idx % len(_DOT_TRAIL)]
        t = f"{elapsed:.0f}s" if elapsed >= 10 else f"{elapsed:.1f}s"
        return f"\r  {colour}{dot}{RESET} {MUTED}{self.action}{RESET} {FAINT}{t}{RESET}  "

    def _render_separator_frame(self, idx: int) -> str:
        """Build one frame of the animated separator pulse."""
        w = term_width() - 4
        pulse = _SEPARATOR_PULSE
        p_len = len(pulse)
        parts = []
        for col in range(w):
            ci = col - (idx % (w + p_len) - p_len)
            if 0 <= ci < p_len:
                parts.append(f"\033[38;5;{pulse[ci]}m─")
            else:
                parts.append(f"{FAINT}─")
        return "  " + "".join(parts) + RESET

    def _animate(self) -> None:
        idx = 0
        interval = 1.0 / self.FPS
        is_tty = sys.stdout.isatty()
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_time
            spinner_line = self._render_frame(idx, elapsed)

            if is_tty:
                # Animate the separator one line above the spinner
                sep_frame = self._render_separator_frame(idx)
                sys.stdout.write(
                    f"\r\033[1A"      # move to start of line, up 1 line
                    f"{sep_frame}"    # draw separator
                    f"\r\033[1B"      # move to start of line, down 1 line
                    f"{spinner_line}" # draw spinner
                )
            else:
                sys.stdout.write(spinner_line)
            sys.stdout.flush()
            idx += 1
            self._stop_event.wait(interval)

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        elapsed = time.monotonic() - self._start_time
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        # Freeze the separator to a static faint rule
        w = term_width() - 4
        if sys.stdout.isatty():
            sys.stdout.write(
                f"\r\033[1A"
                f"  {FAINT}{'─' * w}{RESET}"
                f"\r\033[1B"
            )

        # Print final checkmark and leave it on screen.
        # Pad with spaces to overwrite any leftover spinner text.
        t = f"{elapsed:.1f}s"
        sys.stdout.write(f"\r  {GREEN}✓{RESET} {MUTED}{self.action}{RESET} {FAINT}{t}{RESET}        ")
        sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


@contextmanager
def spinner(label: str = "Working", style: str | None = None) -> Generator[Spinner, None, None]:
    """Context manager for the animated spinner.

    Example::

        with theme.spinner("Thinking"):
            response = client.chat(messages)
    """
    s = Spinner(label, style)
    s.start()
    try:
        yield s
    finally:
        s.stop()
