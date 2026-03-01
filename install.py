#!/usr/bin/env python3
"""
install.py  —  One-time setup for Whisper app.
Generates whisper.ico from the project favicon SVG and creates a
Start Menu shortcut so you can pin it like a normal app.

Run once: python install.py
"""
import math
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# ── Paths ────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent.resolve()
ICO_OUT     = HERE / "whisper.ico"
SCRIPT      = HERE / "whisper.py"
PYTHONW     = Path(sys.executable).parent / "pythonw.exe"
START_MENU  = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Whisper.lnk"

# ── Icon renderer ─────────────────────────────────────────────────────────────
# Reproduces E:\dev\claude_code\website\Logos\favicon.svg at multiple sizes.
# SVG is 512×512 with:
#   • Rounded-rect background  #2C2C2C  rx=64
#   • Diamond outline          #FFFFFF  stroke-width=36  (miter join)
#   • Chevron                  #4ECDB8  stroke-width=36

def _polygon_outline(draw: ImageDraw.ImageDraw, points, color, width):
    """Draw a closed polygon outline with thick lines and approximate miter joins."""
    n = len(points)
    for i in range(n):
        draw.line([points[i], points[(i + 1) % n]], fill=color, width=width)
    # Fill corners with circles to approximate mitered joins
    r = width // 2
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _polyline(draw: ImageDraw.ImageDraw, points, color, width):
    """Draw an open polyline with thick lines and rounded end-caps."""
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=width)
    r = width // 2
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def render_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size / 512.0

    # Background — rounded rectangle
    rx = max(1, round(64 * s))
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=rx, fill="#2C2C2C")

    sw = max(2, round(36 * s))  # stroke width scaled

    # Diamond:  M 256 80 L 376 200 L 256 320 L 136 200 Z
    diamond = [
        (round(256 * s), round(80 * s)),
        (round(376 * s), round(200 * s)),
        (round(256 * s), round(320 * s)),
        (round(136 * s), round(200 * s)),
    ]
    _polygon_outline(draw, diamond, "#FFFFFF", sw)

    # Chevron:  M 136 272 L 256 392 L 376 272
    chevron = [
        (round(136 * s), round(272 * s)),
        (round(256 * s), round(392 * s)),
        (round(376 * s), round(272 * s)),
    ]
    _polyline(draw, chevron, "#4ECDB8", sw)

    return img


def build_ico():
    print("Building whisper.ico ...")
    # Render at high resolution; Pillow rescales to each requested size
    master = render_icon(256)
    master.save(
        ICO_OUT,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"  -> {ICO_OUT}")


# ── Shortcut creator ──────────────────────────────────────────────────────────
SHORTCUT_PS = r"""
$ws  = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("{lnk}")
$lnk.TargetPath       = "{target}"
$lnk.Arguments        = '"{script}"'
$lnk.WorkingDirectory = "{workdir}"
$lnk.IconLocation     = "{icon},0"
$lnk.Description      = "Whisper — Voice to text"
$lnk.Save()
Write-Host "Shortcut created: {lnk}"
"""

def create_shortcut():
    if not PYTHONW.exists():
        print(f"WARNING: pythonw.exe not found at {PYTHONW}")
        print("         Shortcut will use python.exe (console window will flash briefly)")
        target = sys.executable
    else:
        target = str(PYTHONW)

    ps = SHORTCUT_PS.format(
        lnk     = str(START_MENU),
        target  = target,
        script  = str(SCRIPT),
        workdir = str(HERE),
        icon    = str(ICO_OUT),
    )

    print("Creating Start Menu shortcut ...")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR creating shortcut:")
        print(result.stderr)
    else:
        print(f"  -> {START_MENU}")
        print()
        print("To pin to Start Menu:")
        print("  1. Press Win key and type 'Whisper'")
        print("  2. Right-click the app -> 'Pin to Start'")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_ico()
    create_shortcut()
    print()
    print("Done! Run 'Whisper' from the Start Menu (no terminal window).")
