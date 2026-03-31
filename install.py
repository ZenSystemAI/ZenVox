#!/usr/bin/env python3
"""
install.py  —  One-time setup for ZenVox.
Converts the icon PNG to .ico and creates Start Menu + Startup shortcuts.

Run once: python install.py
"""
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent.resolve()
ICON_PNG    = HERE / "zenvox_logo.png"
ICO_OUT     = HERE / "zenvox.ico"
SCRIPT      = HERE / "zenvox.py"
PYTHONW     = Path(sys.executable).parent / "pythonw.exe"
START_MENU  = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/ZenVox.lnk"


def build_ico():
    if not ICON_PNG.exists():
        print(f"Icon source not found: {ICON_PNG}")
        print("Run the app first or place zenvox_logo.png in the project root")
        return False
    print("Building zenvox.ico from ZenVox logo...")
    img = Image.open(ICON_PNG).convert("RGBA")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ICO_OUT, format="ICO", sizes=sizes)
    print(f"  -> {ICO_OUT}")
    return True


# ── Shortcut creator ──────────────────────────────────────────────────────────
SHORTCUT_PS = r"""
$ws  = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("{lnk}")
$lnk.TargetPath       = "{target}"
$lnk.Arguments        = '"{script}"'
$lnk.WorkingDirectory = "{workdir}"
$lnk.IconLocation     = "{icon},0"
$lnk.Description      = "ZenVox — Voice to text, cleaned by AI"
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
        print("  1. Press Win key and type 'ZenVox'")
        print("  2. Right-click the app -> 'Pin to Start'")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if build_ico():
        create_shortcut()
    print()
    print("Done! Run 'ZenVox' from the Start Menu (no terminal window).")
