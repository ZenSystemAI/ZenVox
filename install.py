#!/usr/bin/env python3
"""
install.py  —  One-command setup for ZenVox.

Creates a venv, installs all dependencies (with GPU auto-detection),
converts the logo to .ico (Windows), and creates desktop integration
(Start Menu shortcut on Windows, .desktop file on Linux).

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

# Platform-specific paths
if sys.platform == "win32":
    VENV_PYTHON  = VENV_DIR / "Scripts" / "python.exe"
    VENV_PYTHONW = VENV_DIR / "Scripts" / "pythonw.exe"
    VENV_PIP     = VENV_DIR / "Scripts" / "pip.exe"
    START_MENU   = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/ZenVox.lnk"
else:
    VENV_PYTHON  = VENV_DIR / "bin" / "python3"
    VENV_PYTHONW = None  # No pythonw equivalent on Linux
    VENV_PIP     = VENV_DIR / "bin" / "pip"
    DESKTOP_FILE = Path.home() / ".local" / "share" / "applications" / "zenvox.desktop"


def _run(cmd, desc=None, timeout=600):
    """Run a command, print output live, raise on failure."""
    if desc:
        print(f"\n  {desc}")
    try:
        result = subprocess.run(cmd, cwd=str(HERE), timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  ERROR: command timed out after {timeout}s")
        sys.exit(1)
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
    print("  No NVIDIA GPU detected - will use CPU for transcription")
    return False


# -- Step 1: Create venv --
def create_venv():
    if VENV_DIR.exists() and VENV_PYTHON.exists():
        print("  .venv already exists, skipping creation")
        return
    print("  Creating virtual environment...")
    venv.create(str(VENV_DIR), with_pip=True, clear=True)
    print(f"  -> {VENV_DIR}")


# -- Step 2: Install dependencies --
def install_deps(has_gpu):
    _run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
         "Upgrading pip...")
    _run([str(VENV_PIP), "install", ".[all]"],
         "Installing core dependencies + all cleaning providers...")
    if has_gpu:
        _run([str(VENV_PIP), "install", ".[gpu]"],
             "Installing GPU/CUDA acceleration (ctranslate2)...")


# -- Step 3: Check Linux system dependencies --
def check_linux_deps():
    """Check for required system packages on Linux."""
    missing = []
    import shutil
    session_type = os.environ.get("XDG_SESSION_TYPE", "x11").lower()
    io_deps = (
        [("wtype", "wtype"), ("wl-copy", "wl-clipboard"), ("wl-paste", "wl-clipboard")]
        if session_type == "wayland"
        else [("xdotool", "xdotool"), ("xclip", "xclip")]
    )
    tray_deps = [("python3", "python3-gi"), ("python3", "gir1.2-ayatanaappindicator3-0.1")]
    deps = io_deps + tray_deps
    for cmd, pkg in io_deps:
        if not shutil.which(cmd):
            missing.append(pkg)
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        try:
            gi.require_version("AppIndicator3", "0.1")
        except ValueError:
            gi.require_version("AyatanaAppIndicator3", "0.1")
    except Exception:
        missing.extend(pkg for _, pkg in tray_deps)
    if missing:
        missing = sorted(set(missing))
        print(f"  Missing system packages for {session_type}: {', '.join(missing)}")
        print(f"  Install with: sudo apt install {' '.join(missing)}")
    else:
        pkg_names = ", ".join(sorted(set(pkg for _, pkg in deps)))
        print(f"  All system dependencies present for {session_type} ({pkg_names})")


# -- Step 4: Build .ico (Windows only) --
def build_ico():
    if not ICON_PNG.exists():
        print(f"  Logo not found: {ICON_PNG}")
        print("  Skipping .ico generation")
        return False
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


# -- Step 5: Desktop integration --
def create_shortcut_windows():
    """Create Windows Start Menu shortcut."""
    target = str(VENV_PYTHONW) if VENV_PYTHONW and VENV_PYTHONW.exists() else str(VENV_PYTHON)
    ps_script = f'''
$ws  = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("{START_MENU}")
$lnk.TargetPath       = "{target}"
$lnk.Arguments        = '"{SCRIPT}"'
$lnk.WorkingDirectory = "{HERE}"
$lnk.IconLocation     = "{ICO_OUT},0"
$lnk.Description      = "ZenVox - Voice to text, cleaned by AI"
$lnk.Save()
'''
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR creating shortcut: {result.stderr.strip()}")
    else:
        print(f"  -> {START_MENU}")


def create_desktop_file_linux():
    """Create .desktop file for Linux application menu."""
    DESKTOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    icon_path = ICON_PNG if ICON_PNG.exists() else ""
    content = f"""[Desktop Entry]
Type=Application
Name=ZenVox
Comment=Voice to text, cleaned by AI
Exec={VENV_PYTHON} {SCRIPT}
Icon={icon_path}
Terminal=false
Categories=AudioVideo;Audio;Utility;
Keywords=voice;speech;transcription;whisper;dictation;
StartupNotify=false
"""
    DESKTOP_FILE.write_text(content, encoding="utf-8")
    print(f"  -> {DESKTOP_FILE}")
    # Update desktop database if available
    import shutil
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(DESKTOP_FILE.parent)],
                       capture_output=True)


# -- Main --
def main():
    print("=" * 50)
    print("  ZenVox - One-Command Setup")
    print("=" * 50)

    print("\n[1/5] Virtual environment")
    create_venv()

    print("\n[2/5] GPU detection")
    has_gpu = _has_nvidia_gpu()

    print("\n[3/5] Installing dependencies")
    install_deps(has_gpu)

    if sys.platform != "win32":
        print("\n[4/5] System dependencies")
        check_linux_deps()
    else:
        print("\n[4/5] System dependencies (Windows - skipped)")

    print("\n[5/5] Desktop integration")
    if sys.platform == "win32":
        ico_ok = build_ico()
        if ico_ok:
            create_shortcut_windows()
        else:
            print("  Skipping shortcut (no .ico available)")
    else:
        create_desktop_file_linux()

    print("\n" + "=" * 50)
    print("  SETUP COMPLETE")
    print("=" * 50)
    print()
    print(f"  Run:  {VENV_PYTHON} zenvox.py")
    if sys.platform == "win32" and VENV_PYTHONW and VENV_PYTHONW.exists():
        print("  Or launch 'ZenVox' from the Start Menu")
    elif sys.platform != "win32":
        print("  Or launch 'ZenVox' from your application menu")
    print()
    if has_gpu:
        print("  GPU acceleration enabled - first run downloads")
        print("  the Whisper model (~1.5 GB).")
    else:
        print("  CPU mode - first run downloads the Whisper model.")
        print("  For faster transcription, install NVIDIA CUDA drivers")
        print("  and re-run this script.")
    print()


if __name__ == "__main__":
    main()
