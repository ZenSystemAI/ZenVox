#!/usr/bin/env python3
"""
install.py  —  One-command setup for ZenVox.

Creates a venv, installs all dependencies (with GPU auto-detection),
converts the logo to .ico, and creates a Start Menu shortcut.

Run once:  python install.py
"""
import os
import subprocess
import sys
import venv
from pathlib import Path

HERE       = Path(__file__).parent.resolve()
VENV_DIR   = HERE / ".venv"
ICON_PNG   = HERE / "zenvox_logo.png"
ICO_OUT    = HERE / "zenvox.ico"
SCRIPT     = HERE / "zenvox.py"

# Windows paths inside the venv
VENV_PYTHON  = VENV_DIR / "Scripts" / "python.exe"
VENV_PYTHONW = VENV_DIR / "Scripts" / "pythonw.exe"
VENV_PIP     = VENV_DIR / "Scripts" / "pip.exe"

START_MENU = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/ZenVox.lnk"


def _run(cmd, desc=None):
    """Run a command, print output live, raise on failure."""
    if desc:
        print(f"\n  {desc}")
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)


def _has_nvidia_gpu():
    """Check if an NVIDIA GPU is available via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip().split("\n")[0]
            print(f"  Detected: {name}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("  No NVIDIA GPU detected — will use CPU for transcription")
    return False


# ── Step 1: Create venv ──────────────────────────────────────────────────────
def create_venv():
    if VENV_DIR.exists() and VENV_PYTHON.exists():
        print("  .venv already exists, skipping creation")
        return
    print("  Creating virtual environment...")
    venv.create(str(VENV_DIR), with_pip=True, clear=True)
    print(f"  -> {VENV_DIR}")


# ── Step 2: Install dependencies ─────────────────────────────────────────────
def install_deps(has_gpu):
    # Upgrade pip first
    _run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
         "Upgrading pip...")

    # Core + all cleaning providers
    _run([str(VENV_PIP), "install", ".[all]"],
         "Installing core dependencies + all cleaning providers...")

    # GPU support (ctranslate2 with CUDA)
    if has_gpu:
        _run([str(VENV_PIP), "install", ".[gpu]"],
             "Installing GPU/CUDA acceleration (ctranslate2)...")


# ── Step 3: Build .ico ───────────────────────────────────────────────────────
def build_ico():
    if not ICON_PNG.exists():
        print(f"  Logo not found: {ICON_PNG}")
        print("  Skipping .ico generation")
        return False
    # Use the venv's Pillow
    code = (
        "from PIL import Image; "
        f"img = Image.open(r'{ICON_PNG}').convert('RGBA'); "
        f"img.save(r'{ICO_OUT}', format='ICO', "
        "sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"
    )
    result = subprocess.run([str(VENV_PYTHON), "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Failed to build .ico: {result.stderr.strip()}")
        return False
    print(f"  -> {ICO_OUT}")
    return True


# ── Step 4: Create Start Menu shortcut ───────────────────────────────────────
def create_shortcut():
    target = str(VENV_PYTHONW) if VENV_PYTHONW.exists() else str(VENV_PYTHON)

    ps_script = f'''
$ws  = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("{START_MENU}")
$lnk.TargetPath       = "{target}"
$lnk.Arguments        = '"{SCRIPT}"'
$lnk.WorkingDirectory = "{HERE}"
$lnk.IconLocation     = "{ICO_OUT},0"
$lnk.Description      = "ZenVox — Voice to text, cleaned by AI"
$lnk.Save()
'''
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR creating shortcut: {result.stderr.strip()}")
    else:
        print(f"  -> {START_MENU}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  ZenVox — One-Command Setup")
    print("=" * 50)

    print("\n[1/4] Virtual environment")
    create_venv()

    print("\n[2/4] GPU detection")
    has_gpu = _has_nvidia_gpu()

    print("\n[3/4] Installing dependencies")
    install_deps(has_gpu)

    print("\n[4/4] Desktop integration")
    ico_ok = build_ico()
    if ico_ok:
        create_shortcut()
    else:
        print("  Skipping shortcut (no .ico available)")

    print("\n" + "=" * 50)
    print("  SETUP COMPLETE")
    print("=" * 50)
    print()
    if VENV_PYTHONW.exists():
        print(f"  Run:  {VENV_PYTHONW} zenvox.py")
    else:
        print(f"  Run:  {VENV_PYTHON} zenvox.py")
    print("  Or launch 'ZenVox' from the Start Menu")
    print()
    if has_gpu:
        print("  GPU acceleration enabled — first run downloads")
        print("  the Whisper model (~1.5 GB).")
    else:
        print("  CPU mode — first run downloads the Whisper model.")
        print("  For faster transcription, install NVIDIA CUDA drivers")
        print("  and re-run this script.")
    print()


if __name__ == "__main__":
    main()
