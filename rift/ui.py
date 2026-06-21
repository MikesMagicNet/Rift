"""
ui.py
=====
Terminal UI layer for Rift: themes, skins, colors, and animations.

Everything here is pure-stdlib ANSI. No third-party deps.
Colors auto-disable when output is not a TTY (pipes, redirects) or when
the NO_COLOR environment variable is set, so logs stay clean.
"""

from dependencies import os, sys, time, threading, Any, Optional


# ---------------------------------------------------------------------------
# Color support detection
# ---------------------------------------------------------------------------

def colors_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("RIFT_FORCE_COLOR") is not None:
        return True
    return sys.stdout.isatty()


# 256-color / truecolor helpers
def fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"


# ---------------------------------------------------------------------------
# Themes / skins
# ---------------------------------------------------------------------------
# Each theme defines colors for: banner, user prompt, model name, model reply,
# reasoning text, system/info, error, accent, and a border glyph set.

THEMES: dict[str, dict[str, Any]] = {
    "neon": {
        "desc": "Cyberpunk neon — magenta/cyan on black",
        "banner": fg(255, 0, 170),
        "user": fg(0, 255, 200),
        "model_name": fg(255, 0, 170),
        "reply": fg(220, 240, 255),
        "reasoning": fg(120, 120, 160),
        "info": fg(0, 200, 255),
        "error": fg(255, 70, 70),
        "accent": fg(255, 220, 0),
        "border": "═",
        "prompt_glyph": "❯",
    },
    "matrix": {
        "desc": "Matrix — shades of green",
        "banner": fg(0, 255, 65),
        "user": fg(120, 255, 120),
        "model_name": fg(0, 255, 65),
        "reply": fg(180, 255, 180),
        "reasoning": fg(0, 120, 40),
        "info": fg(0, 200, 90),
        "error": fg(255, 80, 80),
        "accent": fg(200, 255, 0),
        "border": "─",
        "prompt_glyph": ">",
    },
    "solar": {
        "desc": "Solarized warm — amber/orange",
        "banner": fg(255, 160, 0),
        "user": fg(255, 200, 90),
        "model_name": fg(255, 140, 0),
        "reply": fg(245, 230, 200),
        "reasoning": fg(150, 120, 70),
        "info": fg(200, 170, 60),
        "error": fg(220, 50, 47),
        "accent": fg(133, 153, 0),
        "border": "━",
        "prompt_glyph": "»",
    },
    "ocean": {
        "desc": "Deep ocean — blues and teals",
        "banner": fg(0, 150, 255),
        "user": fg(80, 200, 255),
        "model_name": fg(0, 150, 255),
        "reply": fg(210, 235, 255),
        "reasoning": fg(70, 110, 150),
        "info": fg(0, 180, 220),
        "error": fg(255, 90, 90),
        "accent": fg(0, 255, 210),
        "border": "≈",
        "prompt_glyph": "◆",
    },
    "mono": {
        "desc": "Minimal monochrome — no colors, clean glyphs",
        "banner": "",
        "user": "",
        "model_name": "",
        "reply": "",
        "reasoning": "",
        "info": "",
        "error": "",
        "accent": "",
        "border": "-",
        "prompt_glyph": ">",
    },
}

DEFAULT_THEME = "neon"

SPINNERS: dict[str, list[str]] = {
    "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "line": ["|", "/", "-", "\\"],
    "bounce": ["⠁", "⠂", "⠄", "⠂"],
    "pulse": ["•", "◦", "·", "◦"],
    "arc": ["◜", "◠", "◝", "◞", "◡", "◟"],
}


# ---------------------------------------------------------------------------
# Theme renderer
# ---------------------------------------------------------------------------

class Theme:
    """Resolved theme with helpers that respect the color toggle."""

    def __init__(self, name: str, animations: bool = True,
                 typing_speed: float = 0.0, spinner: str = "dots"):
        self.name = name if name in THEMES else DEFAULT_THEME
        self.t = THEMES[self.name]
        self.color = colors_enabled()
        self.animations = animations and self.color
        self.typing_speed = max(0.0, float(typing_speed))
        self.spinner_frames = SPINNERS.get(spinner, SPINNERS["dots"])

    # color wrappers ------------------------------------------------------
    def _c(self, key: str, text: str, bold: bool = False) -> str:
        if not self.color:
            return text
        code = self.t.get(key, "")
        prefix = (BOLD if bold else "") + code
        return f"{prefix}{text}{RESET}"

    def user(self, s: str) -> str:        return self._c("user", s, bold=True)
    def model_name(self, s: str) -> str:  return self._c("model_name", s, bold=True)
    def reply(self, s: str) -> str:       return self._c("reply", s)
    def reasoning(self, s: str) -> str:   return self._c("reasoning", s)
    def info(self, s: str) -> str:        return self._c("info", s)
    def error(self, s: str) -> str:       return self._c("error", s, bold=True)
    def accent(self, s: str) -> str:      return self._c("accent", s, bold=True)
    def banner_c(self, s: str) -> str:    return self._c("banner", s, bold=True)

    @property
    def prompt_glyph(self) -> str:
        return self.t.get("prompt_glyph", ">")

    @property
    def border_char(self) -> str:
        return self.t.get("border", "-")

    def rule(self, width: int = 60) -> str:
        return self.banner_c(self.border_char * width)

    # streaming print with optional typewriter effect --------------------
    def stream_print(self, text: str, kind: str = "reply") -> None:
        colored = getattr(self, kind, self.reply)(text) if self.typing_speed == 0 else None
        if self.typing_speed > 0 and self.color:
            for chunk in text:
                sys.stdout.write(getattr(self, kind, self.reply)(chunk))
                sys.stdout.flush()
                time.sleep(self.typing_speed)
        else:
            sys.stdout.write(colored if colored is not None else text)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Spinner animation (runs in a background thread while waiting on the API)
# ---------------------------------------------------------------------------

class Spinner:
    """Animated 'thinking' spinner shown while waiting for the first token."""

    def __init__(self, theme: Theme, label: str = "thinking"):
        self.theme = theme
        self.label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _spin(self) -> None:
        frames = self.theme.spinner_frames
        i = 0
        while not self._stop.is_set():
            frame = frames[i % len(frames)]
            line = f"\r{self.theme.accent(frame)} {self.theme.info(self.label)}…"
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1
        # clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.label) + 6) + "\r")
        sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        if self.theme.animations:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

RIFT_ASCII = r"""
  ____  _  __ _
 |  _ \(_)/ _| |_
 | |_) | | |_| __|
 |  _ <| |  _| |_
 |_| \_\_|_|  \__|
"""


def animate_banner(theme: Theme) -> None:
    """Print the Rift banner with an optional line-by-line fade-in."""
    lines = RIFT_ASCII.strip("\n").splitlines()
    if theme.animations:
        for ln in lines:
            print(theme.banner_c(ln))
            time.sleep(0.05)
    else:
        for ln in lines:
            print(theme.banner_c(ln))


def print_header(theme: Theme, model: str, show_banner: bool = True) -> None:
    width = 62
    
    # Draw top border
    print(theme.accent("┌" + "─" * (width - 2) + "┐"))
    
    # R I F T   C H A T title centered
    title = "R I F T   C H A T"
    title_len = len(title)
    pad_left = (width - 2 - title_len) // 2
    pad_right = width - 2 - title_len - pad_left
    print(theme.accent("│") + " " * pad_left + theme.banner_c(title) + " " * pad_right + theme.accent("│"))
    
    # Separator
    print(theme.accent("├" + "─" * (width - 2) + "┤"))
    
    # Model info line
    model_lbl = "  Model:  "
    model_val = model
    # Truncate model name if it's too long
    max_val_len = width - 4 - len(model_lbl)
    if len(model_val) > max_val_len:
        model_val = model_val[:max_val_len - 3] + "..."
    plain_len = len(model_lbl) + len(model_val)
    pad = " " * (width - 2 - plain_len)
    print(theme.accent("│") + theme.info(model_lbl) + theme.model_name(model_val) + pad + theme.accent("│"))
    
    # Theme & commands info line
    cmds_lbl = "  Cmds:   "
    cmds_val = "/model · theme <name> · themes · clear · exit"
    plain_len_cmds = len(cmds_lbl) + len(cmds_val)
    pad_cmds = " " * (width - 2 - plain_len_cmds)
    print(theme.accent("│") + theme.info(cmds_lbl) + theme.accent(cmds_val) + pad_cmds + theme.accent("│"))
    
    # Draw bottom border
    print(theme.accent("└" + "─" * (width - 2) + "┘"))
    print()


def list_themes(theme: Theme) -> None:
    print()
    for name, data in THEMES.items():
        marker = theme.accent(" (active)") if name == theme.name else ""
        sample = Theme(name)
        swatch = sample.banner_c("████") if sample.color else "----"
        print(f"  {swatch}  {sample.accent(name):<22}{marker}")
        print(f"        {theme.info(data['desc'])}")
    print()
