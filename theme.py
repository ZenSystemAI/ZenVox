"""
theme.py — ZenVox design tokens (ZenSEO design language, dark-first).

One palette for the whole app: main window, floating overlay, and tray icons
all read from here. Derived from the ZenSEO design contract: teal #4ECDB8
brand accent used sparingly, zinc-with-a-teal-hint neutral ramp, hairline
borders, dark sidebar, mono numerals.

Fonts resolve once at import against what fontconfig actually has installed —
the old hardcoded ("Inter", …) tuples silently fell back to the Tk default
because Inter was never installed on this machine.
"""
import subprocess
import sys

# ── Surfaces (zinc ramp with a subtle teal hint, dark-first) ─────────────────
BG        = "#0D100F"   # window / content background
PANEL     = "#151918"   # cards, sidebar-adjacent surfaces
PANEL2    = "#1B201E"   # inputs, nested surfaces ("muted")
SIDEBAR   = "#0A0C0B"   # always-dark nav rail
BORDER    = "#242927"   # hairline card/control borders
BORDER_SOFT = "#1C211F" # row dividers, quieter than BORDER

# ── Text ─────────────────────────────────────────────────────────────────────
TEXT      = "#F2F4F3"   # primary
MUTED     = "#9CA3A0"   # secondary / labels
FAINT     = "#5C6360"   # tertiary / hints

# ── Brand (ONE accent — active states, key metrics, primary actions) ─────────
BRAND       = "#4ECDB8"
BRAND_HOVER = "#3DB8A3"
BRAND_TEXT  = "#78DAC0"   # brighter teal for small text on dark (AA)
BRAND_DIM   = "#12352D"   # brand-tinted fill (selection pills, tints)
ON_BRAND    = "#0B1210"   # dark text on brand fills

# ── State colors (recording / pipeline / error) ──────────────────────────────
REC       = "#EF5350"   # recording red
REC_HOVER = "#D32F2F"
REC_DIM   = "#3A1B1A"   # red-tinted surface while recording
AMBER     = "#FFB74D"   # transcribing / cleaning
AMBER_DIM = "#3A2E17"
ERROR     = "#EF5350"
OK        = BRAND

# Tray dot colors keyed by app state (idle dot == brand teal, no more drift)
TRAY = {
    "idle":         BRAND,
    "recording":    REC,
    "transcribing": AMBER,
    "loading":      "#9E9E9E",
    "error":        "#B71C1C",
}

# ── Radii vocabulary (ZenSEO: cards xl, controls lg, chips md) ───────────────
R_CARD  = 14
R_CTRL  = 8
R_CHIP  = 6
R_PILL  = 16

# ── Font resolution ──────────────────────────────────────────────────────────
_SANS_PREFS = ["Mona Sans", "Inter Tight", "Inter", "Noto Sans", "Ubuntu", "DejaVu Sans"]
_SANS_MED_PREFS = ["Mona Sans Medium", "Mona Sans", "Inter Tight", "Inter", "Noto Sans Medium", "Noto Sans"]
_SANS_SEMI_PREFS = ["Mona Sans SemiBold", "Mona Sans", "Inter Tight", "Inter", "Noto Sans"]
_MONO_PREFS = ["Geist Mono", "JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "Ubuntu Mono"]


def _installed_families():
    """Font families fontconfig knows about (Linux). Empty set on failure —
    callers fall back to the first preference and let Tk substitute."""
    if sys.platform == "win32":
        return set()
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True,
                             text=True, timeout=3)
        fams = set()
        for line in out.stdout.splitlines():
            for fam in line.split(","):
                fams.add(fam.strip())
        return fams
    except Exception:
        return set()


_FAMILIES = _installed_families()


def _pick(prefs, fallback):
    for name in prefs:
        if not _FAMILIES or name in _FAMILIES:
            return name
    return fallback


SANS      = _pick(_SANS_PREFS, "TkDefaultFont")
SANS_MED  = _pick(_SANS_MED_PREFS, SANS)
SANS_SEMI = _pick(_SANS_SEMI_PREFS, SANS)
MONO      = _pick(_MONO_PREFS, "TkFixedFont")


def font(size, weight="normal", mono=False):
    """Font tuple for CTk widgets. weight: normal|medium|semibold|bold.
    Medium/semibold map to dedicated families when installed (variable-font
    statics), falling back to synthetic bold."""
    if mono:
        return (MONO, size, "bold") if weight in ("semibold", "bold") else (MONO, size)
    if weight == "bold":
        return (SANS, size, "bold")
    if weight == "semibold":
        return (SANS_SEMI, size) if SANS_SEMI != SANS else (SANS, size, "bold")
    if weight == "medium":
        return (SANS_MED, size) if SANS_MED != SANS else (SANS, size)
    return (SANS, size)


# ── Type scale (ZenSEO rhythm) ───────────────────────────────────────────────
F_TITLE   = font(21, "semibold")   # page title
F_SECTION = font(13, "semibold")   # section header
F_BODY    = font(13)               # body / inputs
F_LABEL   = font(12)               # form labels
F_SMALL   = font(11)               # hints, meta
F_TINY    = font(10)               # chips
F_METRIC  = font(22, "semibold", mono=True)   # stat values
F_MONO_S  = font(11, mono=True)    # timestamps, durations
