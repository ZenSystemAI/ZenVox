#!/usr/bin/env python3
"""
zenvox.py  —  ZenVox: Voice to text, cleaned by AI.
media_play_pause = record (auto-stops on silence)
Ctrl+Alt+F11 = re-paste last transcription
"""
import os
import sys
import concurrent.futures
import io
import queue
import shutil
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass
from enum import Enum

# Suppress pythonw.exe crashes from writing to missing stderr/stdout (Windows only)
if sys.platform == "win32" and sys.executable.lower().endswith("pythonw.exe"):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

warnings.filterwarnings("ignore", module="google.genai")


def _prefer_linux_appindicator_backend():
    """Expose Ubuntu GI packages to the venv and prefer AppIndicator when available."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("PYSTRAY_BACKEND"):
        return

    dist_packages = "/usr/lib/python3/dist-packages"
    if os.path.isdir(dist_packages) and dist_packages not in sys.path:
        sys.path.append(dist_packages)

    try:
        import gi
        gi.require_version("Gtk", "3.0")
        try:
            gi.require_version("AppIndicator3", "0.1")
        except ValueError:
            gi.require_version("AyatanaAppIndicator3", "0.1")
        os.environ["PYSTRAY_BACKEND"] = "appindicator"
    except Exception:
        pass


_prefer_linux_appindicator_backend()

import customtkinter as ctk
import numpy as np
import pyautogui
import pyperclip
import pystray
import sounddevice as sd
from datetime import datetime

from config import (
    Settings, SETTINGS_FILE,
    SAMPLE_RATE, CPU_THREADS, NUM_WORKERS,
    MODELS, LANGS, CLEANING_PRESETS, OUTPUT_MODES, ICONS,
    RAW_PRESET, PRESET_NAMES, CAPTURE_MODES, CAPTURE_TOGGLE, CAPTURE_PTT,
    TRANSCRIPTION_BACKENDS, BACKEND_LOCAL, BACKEND_REMOTE,
    DEVICE, COMPUTE, DEVICE_LABEL, BEEP_START, BEEP_STOP,
    NO_SPEECH_PEAK_FLOOR, MAX_RECORD_SECONDS, STUCK_PIPELINE_TIMEOUT,
    setup_logging, list_input_devices, detect_ui_scale, APP_DIR,
)
from providers import PROVIDERS, PROVIDER_NAMES, create_provider, translation_detected
from history import History
from dictionary import Dictionary, DictionaryEntry
import theme as T

log = setup_logging()


def _patch_pystray_appindicator_icons():
    """Teach pystray's AppIndicator backend to use a real icon name + theme path.

    Reaches into pystray internals — wrapped by the caller so an internals
    drift in a future pystray release degrades the tray icon instead of
    crashing the app at import time.
    """
    if pystray.Icon.__module__ != "pystray._appindicator":
        return
    try:
        import tempfile
        from pystray import _appindicator as _pi
    except Exception:
        return
    if getattr(_pi.Icon, "_zenvox_icon_patch", False):
        return

    def _update_fs_icon(self):
        fd, path = tempfile.mkstemp(prefix="zenvox-tray-", suffix=".png")
        os.close(fd)
        with open(path, "wb") as f:
            self.icon.save(f, "PNG")
        self._icon_path = path
        self._icon_name = os.path.splitext(os.path.basename(path))[0]
        self._icon_valid = True

    def _apply_icon(self):
        icon_path = getattr(self, "_icon_path", None)
        icon_name = getattr(self, "_icon_name", None)
        if not icon_path or not icon_name:
            return
        self._appindicator.set_icon_theme_path(os.path.dirname(icon_path))
        self._appindicator.set_icon_full(icon_name, self.title or self.name)

    def _show(self):
        if self.icon:
            self._remove_fs_icon()
            self._update_fs_icon()
            _apply_icon(self)
        self._appindicator.set_menu(
            self._menu_handle or self._create_default_menu())
        self._appindicator.set_title(self.title)
        self._appindicator.set_status(_pi.AppIndicator.IndicatorStatus.ACTIVE)

    def _update_icon(self):
        self._remove_fs_icon()
        self._update_fs_icon()
        _apply_icon(self)

    def _finalize(self):
        try:
            self._appindicator.set_status(_pi.AppIndicator.IndicatorStatus.PASSIVE)
        except Exception:
            pass
        _pi.GtkIcon._finalize(self)
        del self._appindicator

    _pi.GtkIcon._update_fs_icon = _update_fs_icon
    _pi.Icon._show = _pi.mainloop(_show)
    _pi.Icon._update_icon = _pi.mainloop(_update_icon)
    _pi.Icon._finalize = _finalize
    _pi.Icon._zenvox_icon_patch = True


try:
    _patch_pystray_appindicator_icons()
except Exception:
    log.exception("pystray AppIndicator patch failed — tray icon may not render")


class AppState(str, Enum):
    LOADING = "loading"
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    CLEANING = "cleaning"
    ERROR = "error"


@dataclass
class CleanResult:
    text: str
    used_fallback: bool = False
    reason: str = ""


class TranscriptionError(RuntimeError):
    """Transcription failed (model error, remote unreachable, …). Distinct from
    'no speech detected' so the app can tell the user their dictation was lost
    instead of silently flashing back to Ready."""


@dataclass
class ClipboardState:
    clipboard: str = ""
    primary: str | None = None


@dataclass
class ClipboardOwner:
    process: object | None = None
    selection: str = ""


@dataclass
class ActiveWindowInfo:
    window_id: str = ""
    window_class: str = ""
    window_name: str = ""


_PERSISTENT_SELECTION_OWNERS = {}


def _linux_session_type():
    return os.environ.get("XDG_SESSION_TYPE", "x11").lower()


def _run_text_command(cmd, input_text=None, timeout=2):
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _copy_selection_linux(text, selection):
    owner = _start_selection_owner_linux(text, selection)
    if owner is not None:
        previous = _PERSISTENT_SELECTION_OWNERS.get(selection)
        _close_clipboard_owner(previous)
        _PERSISTENT_SELECTION_OWNERS[selection] = owner
        return True
    return False


def _read_selection_linux(selection, timeout=5):
    session_type = _linux_session_type()
    args = None
    if session_type == "wayland" and shutil.which("wl-paste"):
        args = ["wl-paste", "--no-newline"]
        if selection == "primary":
            args.append("--primary")
    elif shutil.which("xclip"):
        args = ["xclip", "-selection", selection, "-o"]
    elif shutil.which("xsel"):
        flag = "--clipboard" if selection == "clipboard" else "--primary"
        args = ["xsel", flag, "--output"]
    if args is None:
        return ""
    try:
        # Default 5s: X11 selection reads can block when the owner is busy or
        # the display is under load. Off-critical-path snapshots pass a tighter
        # timeout and treat a miss as "nothing to restore".
        result = _run_text_command(args, timeout=timeout)
        return result.stdout if result.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        log.warning(f"Clipboard read timed out ({args[0]} {selection}) — continuing with empty state")
        return ""
    except Exception as e:
        log.warning(f"Clipboard read failed ({args[0]} {selection}): {e}")
        return ""


def _read_clipboard_state(timeout=5):
    if sys.platform == "win32":
        try:
            return ClipboardState(clipboard=pyperclip.paste())
        except Exception:
            return ClipboardState()

    clipboard = _read_selection_linux("clipboard", timeout=timeout)
    primary = None
    if sys.platform.startswith("linux"):
        primary = _read_selection_linux("primary", timeout=timeout)
    if not clipboard:
        try:
            clipboard = pyperclip.paste()
        except Exception:
            clipboard = ""
    return ClipboardState(clipboard=clipboard, primary=primary)


def _close_clipboard_owner(owner):
    if owner is None:
        return
    proc = owner.process if isinstance(owner, ClipboardOwner) else None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


def _cleanup_clipboard_owners(owners):
    for owner in owners or ():
        _close_clipboard_owner(owner)


def _start_selection_owner_linux(text, selection):
    session_type = _linux_session_type()
    args = None
    if session_type == "wayland" and shutil.which("wl-copy"):
        args = ["wl-copy", "--foreground"]
        if selection == "primary":
            args.append("--primary")
    elif shutil.which("xclip"):
        args = ["xclip", "-selection", selection, "-in"]
    elif shutil.which("xsel"):
        flag = "--clipboard" if selection == "clipboard" else "--primary"
        args = ["xsel", flag, "--input"]
    if args is None:
        return None

    proc = None
    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if proc.stdin is not None:
            proc.stdin.write(text)
            proc.stdin.close()
        time.sleep(0.05 if session_type == "x11" else 0.02)
        if proc.poll() not in (None, 0):
            return None
        return ClipboardOwner(process=proc, selection=selection)
    except Exception:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        return None


def _copy_text(text, include_primary=False, for_immediate_paste=False):
    if sys.platform == "win32":
        pyperclip.copy(text)
        return []

    owners = []
    if for_immediate_paste and sys.platform.startswith("linux"):
        owner = _start_selection_owner_linux(text, "clipboard")
        if owner is not None:
            owners.append(owner)
        else:
            try:
                pyperclip.copy(text)
            except Exception:
                _copy_selection_linux(text, "clipboard")
        if include_primary:
            owner = _start_selection_owner_linux(text, "primary")
            if owner is not None:
                owners.append(owner)
            else:
                _copy_selection_linux(text, "primary")
        return owners

    try:
        pyperclip.copy(text)
    except Exception:
        _copy_selection_linux(text, "clipboard")
    if include_primary:
        _copy_selection_linux(text, "primary")
    return owners


def _restore_clipboard_state(previous, expected_text, last_pasted_text="", timeout=1.5):
    # Tight timeout by default: this runs holding _clipboard_lock, so a hung
    # clipboard owner's read must not block the next dictation for 5s+.
    current = _read_clipboard_state(timeout=timeout)
    if current.clipboard == expected_text and previous.clipboard != last_pasted_text:
        _copy_text(previous.clipboard, include_primary=False)
    if (
        sys.platform.startswith("linux")
        and previous.primary is not None
        and current.primary == expected_text
    ):
        _copy_selection_linux(previous.primary, "primary")


_TERMINAL_CLASS_MARKERS = (
    "gnome-terminal",
    "tilix",
    "xfce4-terminal",
    "konsole",
    "kitty",
    "alacritty",
    "wezterm",
    "terminator",
    "xterm",
    "lxterminal",
    "guake",
    "tilda",
    "rxvt",
)


def _is_terminal_window(window):
    # Match WM_CLASS only — matching the window TITLE flagged ordinary windows
    # (a browser tab named "kitty vs alacritty", an editor with xterm.c open)
    # as terminals, sending them Ctrl+Shift+V instead of Ctrl+V.
    if not isinstance(window, ActiveWindowInfo):
        return False
    return any(marker in window.window_class.lower()
               for marker in _TERMINAL_CLASS_MARKERS)


def _get_window_class(window_id):
    if not window_id or not shutil.which("xprop"):
        return ""
    try:
        result = subprocess.run(
            ["xprop", "-id", window_id, "WM_CLASS"],
            capture_output=True, text=True, timeout=2, check=False)
        if result.returncode != 0:
            return ""
        _, _, raw = result.stdout.partition("=")
        return raw.replace('"', "").strip().lower()
    except Exception:
        return ""


def _get_window_name(window_id):
    if not window_id or not shutil.which("xdotool"):
        return ""
    try:
        result = subprocess.run(
            ["xdotool", "getwindowname", window_id],
            capture_output=True, text=True, timeout=2, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _get_active_window_id_fast():
    """Focused window id only — a single subprocess (~10ms). Used at record
    start where every millisecond delays the mic opening."""
    if sys.platform == "win32" or _linux_session_type() == "wayland":
        return ActiveWindowInfo()
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return ActiveWindowInfo(window_id=result.stdout.strip())
    except Exception:
        pass
    return ActiveWindowInfo()


def _fill_window_details(info):
    """Fill class/name on an ActiveWindowInfo in place (background thread).
    Needed by clean/paste time (terminal detection, per-app rules) — long
    after recording has started."""
    if not isinstance(info, ActiveWindowInfo) or not info.window_id:
        return
    info.window_class = _get_window_class(info.window_id)
    info.window_name = _get_window_name(info.window_id)


def _get_active_window_info():
    """Get the currently focused window metadata on Linux/X11 (synchronous)."""
    info = _get_active_window_id_fast()
    _fill_window_details(info)
    return info


def _simulate_paste(window=None):
    """Simulate paste on the current platform."""
    if sys.platform == "win32":
        pyautogui.hotkey("ctrl", "v")
        return

    session_type = _linux_session_type()
    if session_type == "wayland":
        if shutil.which("wtype"):
            # -k presses the NAMED key; without it wtype treats "Insert" as
            # literal text and types the word into the target window.
            subprocess.run(["wtype", "-M", "shift", "-k", "Insert", "-m", "shift"],
                           check=False)
            return
        log.warning("Wayland paste: wtype not found — cannot inject keystroke")
        pyautogui.hotkey("ctrl", "v")
        return

    # X11
    window_id = window.window_id if isinstance(window, ActiveWindowInfo) else window
    is_terminal = _is_terminal_window(window)
    if window_id:
        try:
            # Skip the activate/focus/settle dance when the target is already
            # focused (the common case — user kept working in the same window):
            # saves 2 subprocess spawns + 100ms on every paste.
            current = subprocess.run(["xdotool", "getactivewindow"],
                                     capture_output=True, text=True, timeout=1, check=False)
            already_active = (current.returncode == 0
                              and current.stdout.strip() == window_id)
            if not already_active:
                subprocess.run(["xdotool", "windowactivate", "--sync", window_id],
                               timeout=2, check=False)
                subprocess.run(["xdotool", "windowfocus", "--sync", window_id],
                               timeout=2, check=False)
                time.sleep(0.1)
        except subprocess.TimeoutExpired:
            log.warning(f"Paste: xdotool windowactivate/focus timed out (wid={window_id})")
        except Exception as e:
            log.warning(f"Paste: xdotool focus failed: {e}")
    else:
        log.warning("Paste: no target window_id — pasting into whatever has focus")

    # Terminals on Ubuntu usually need Ctrl+Shift+V; most other apps expect Ctrl+V.
    key_combo = "ctrl+shift+v" if is_terminal else "ctrl+v"
    if shutil.which("xdotool"):
        result = subprocess.run(
            ["xdotool", "key", "--clearmodifiers", key_combo],
            capture_output=True, text=True, timeout=2, check=False)
        if result.returncode != 0:
            log.warning(f"Paste: xdotool key returned {result.returncode}: {result.stderr.strip()}")
    else:
        pyautogui.hotkey(*key_combo.split("+"))
    log.info(f"Paste sent ({key_combo}) to window={window_id or 'unknown'} terminal={is_terminal}")


# ═══════════════════════════════════════════════════════════════════════════════
# REMOTE ASR — encode + POST to an OpenAI-compatible transcription endpoint
# ═══════════════════════════════════════════════════════════════════════════════
def _audio_to_wav_bytes(audio, sample_rate=SAMPLE_RATE):
    """16kHz float32 mono → 16-bit PCM WAV bytes (for the multipart upload)."""
    import wave
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _post_multipart_audio(url, wav_bytes, fields, timeout=60.0, token=""):
    """Minimal multipart/form-data POST (no requests dependency). Returns parsed JSON."""
    import json as _json
    from urllib import request as _req
    boundary = "----zenvox" + os.urandom(12).hex()
    pre = []
    for name, value in fields.items():
        pre.append(f"--{boundary}\r\n".encode())
        pre.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        pre.append(f"{value}\r\n".encode())
    pre.append(f"--{boundary}\r\n".encode())
    pre.append(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
    pre.append(b"Content-Type: audio/wav\r\n\r\n")
    body = b"".join(pre) + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["X-ZenVox-Key"] = token
    req = _req.Request(url, data=body, method="POST", headers=headers)
    with _req.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE — Recording, transcription, cleaning. Thread-safe.
# ═══════════════════════════════════════════════════════════════════════════════
class ZenVoxEngine:
    def __init__(self, settings):
        self.settings = settings
        self.dictionary = Dictionary.load()
        self._lock = threading.Lock()
        self._whisper_model = None
        self._loaded_model_name = None
        self._vad_model = None
        self._cleaning_provider = None
        self._cleaning_provider_key = None
        self._recording = False
        self._transcribing = False
        self._audio_chunks = []
        self._stream = None
        self._record_start = 0.0
        self._audio_level = 0.0
        self._speech_detected = False
        self._silence_start = None
        self._last_toggle = 0.0
        self._vad_h = self._vad_c = self._vad_context = None
        self._on_vad_stop = None
        self._vad_queue = None
        self._vad_worker_thread = None
        self._native_rate = SAMPLE_RATE
        self._stop_requested = False
        self._vad_autostop = True  # False in push-to-talk (release controls stop)
        self._infer_lock = threading.Lock()  # serialize model use (preview vs final)
        self._on_partial = None
        self._preview_thread = None
        # stop_recording spends seconds joining workers AFTER clearing
        # _recording; without this flag a new start during that window gets its
        # fresh stream closed and its audio wiped by the in-flight stop.
        self._stop_in_progress = False
        # Bumped per recording session; orphaned preview/VAD workers from a
        # previous session compare against it and exit instead of corrupting
        # the new session's state.
        self._session_gen = 0
        self.active_device_label = DEVICE_LABEL

    @property
    def is_recording(self):
        with self._lock:
            return self._recording

    @property
    def is_transcribing(self):
        with self._lock:
            return self._transcribing

    def force_reset(self):
        """Recovery hook: fully release recording/transcribing so a fresh start
        always works. Clearing only the flag would leave a live InputStream and
        VAD worker held — the next start_recording would no-op and the hotkey
        would stay dead."""
        with self._lock:
            self._recording = False
            self._transcribing = False
            self._stop_in_progress = False
            self._session_gen += 1  # orphan any workers still holding the old gen
        try:
            self._stop_vad_worker()
        except Exception as e:
            log.error(f"force_reset vad worker: {e}")
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            log.error(f"force_reset stream close: {e}")

    def _request_stop(self):
        """One-shot gate shared by the VAD-silence and max-duration stop paths.
        Returns True exactly once per recording so _on_vad_stop never double-fires."""
        with self._lock:
            if self._stop_requested or not self._recording:
                return False
            self._stop_requested = True
        return True

    def _fire_stop(self):
        """Trigger the auto-stop callback at most once, safely from any thread
        (audio callback or VAD worker). Never raises into the realtime path."""
        if not self._request_stop():
            return
        cb = self._on_vad_stop
        if cb:
            try:
                cb()
            except Exception as e:
                log.error(f"on_vad_stop failed: {e}")

    @property
    def model_loaded(self):
        with self._lock:
            return self._whisper_model is not None

    @property
    def is_ready(self):
        """Ready to record? Local needs the Whisper model; remote needs only VAD."""
        with self._lock:
            if self.settings.transcription_backend == BACKEND_REMOTE:
                return self._vad_model is not None
            return self._whisper_model is not None

    @property
    def loaded_model_name(self):
        with self._lock:
            return self._loaded_model_name

    @property
    def is_busy(self):
        with self._lock:
            return self._recording or self._transcribing

    @property
    def recording_duration(self):
        return time.time() - self._record_start if self._recording else 0.0

    @property
    def audio_level(self):
        return self._audio_level

    # ── Model ─────────────────────────────────────────────────────────────
    def load_model(self, model_name=None):
        from faster_whisper import WhisperModel
        from faster_whisper.vad import get_vad_model
        target_model = model_name or self.settings.model_name

        # Remote backend: transcription happens on the server, so don't load a
        # local Whisper model at all (saves the local GPU/CPU). Still load VAD —
        # silence auto-stop runs locally and is tiny (ONNX on CPU).
        if self.settings.transcription_backend == BACKEND_REMOTE:
            vad_model = get_vad_model()
            host = self._remote_host()
            reachable = self.remote_reachable(timeout=3.0)
            label = f"Remote ({host})" if reachable else f"Remote ({host}) — UNREACHABLE"
            with self._lock:
                self._whisper_model = None
                self._vad_model = vad_model
                self._loaded_model_name = "(remote)"
                self._cleaning_provider = None
                self.active_device_label = label
            self._get_cleaning_provider()
            if not reachable:
                log.warning(f"Remote ASR at {self.settings.remote_url} is not answering "
                            f"— dictations will fail until it's back (or switch to local)")
            log.info(f"Remote backend active ({label}) — local Whisper not loaded")
            return

        def _build(device, compute):
            return WhisperModel(
                target_model, device=device, compute_type=compute,
                cpu_threads=CPU_THREADS, num_workers=NUM_WORKERS)

        device, compute, label = DEVICE, COMPUTE, DEVICE_LABEL
        try:
            whisper_model = _build(device, compute)
        except Exception as e:
            # The GPU is often shared (vLLM / ComfyUI hold VRAM) — a CUDA OOM at
            # load must NOT kill the app. Fall back to CPU so dictation still
            # works; the user can reload onto GPU later when VRAM frees.
            if device == "cuda":
                log.warning(f"GPU model load failed ({e}); falling back to CPU")
                device, compute, label = "cpu", "int8", "CPU (GPU busy)"
                whisper_model = _build(device, compute)
            else:
                raise
        # Warmup: the first decode pays CUDA context / cuBLAS / memory-pool
        # init (0.5-3s). Absorb it here on the loader thread instead of inside
        # the user's first visible "Transcribing..." wait.
        try:
            warm_start = time.time()
            segs, _ = whisper_model.transcribe(
                np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32), beam_size=1)
            for _ in segs:
                pass
            log.info(f"Warmup decode: {time.time() - warm_start:.2f}s")
        except Exception as e:
            log.warning(f"Warmup decode failed (non-fatal): {e}")
        vad_model = get_vad_model()
        with self._lock:
            self._whisper_model = whisper_model
            self._vad_model = vad_model
            self._loaded_model_name = target_model
            self._cleaning_provider = None
            self.active_device_label = label
        self._get_cleaning_provider()
        log.info(f"Loaded {target_model} on {label}")

    def _remote_host(self):
        try:
            from urllib.parse import urlparse
            return urlparse(self.settings.remote_url).hostname or "remote"
        except Exception:
            return "remote"

    def remote_reachable(self, timeout=2.0):
        """Probe the remote ASR server. Any HTTP answer (even 404) proves the
        server is up — /health exists on server.py but not necessarily on
        whisper.cpp's whisper-server."""
        from urllib import error as _err, request as _req
        from urllib.parse import urlparse
        try:
            u = urlparse(self.settings.remote_url)
            origin = f"{u.scheme}://{u.netloc}"
        except Exception:
            return False
        try:
            with _req.urlopen(origin + "/health", timeout=timeout):
                return True
        except _err.HTTPError:
            return True  # server answered — reachable, just no /health route
        except Exception:
            return False

    # ── Toggle guard ──────────────────────────────────────────────────────
    def request_toggle(self):
        """Atomically decide what a toggle press should do: 'start', 'stop', or
        'reject'. Debounce check, readiness check, and the start/stop decision
        all happen under one lock so two near-simultaneous triggers (GUI button
        + hotkey) can never both decide to start."""
        now = time.time()
        remote = self.settings.transcription_backend == BACKEND_REMOTE
        with self._lock:
            if now - self._last_toggle < 0.4:
                return "reject"
            not_ready = self._vad_model is None if remote else self._whisper_model is None
            if not_ready or self._transcribing or self._stop_in_progress:
                return "reject"
            self._last_toggle = now
            return "stop" if self._recording else "start"

    # ── Recording ─────────────────────────────────────────────────────────
    def start_recording(self, device_id=None, on_vad_stop=None, vad_autostop=True,
                        on_partial=None):
        with self._lock:
            if self._recording or self._stop_in_progress:
                return
            self._recording = True
            self._session_gen += 1
            session_gen = self._session_gen
        self._audio_chunks = []
        self._speech_detected = False
        self._silence_start = None
        self._record_start = time.time()
        self._audio_level = 0.0
        self._on_vad_stop = on_vad_stop
        self._on_partial = on_partial
        self._preview_thread = None
        self._stop_requested = False
        # Push-to-talk: key release stops, so don't auto-stop on VAD silence
        # (the max-duration safety cap still fires via the audio callback).
        self._vad_autostop = vad_autostop
        self._vad_h = np.zeros((1, 1, 128), dtype="float32")
        self._vad_c = np.zeros((1, 1, 128), dtype="float32")
        self._vad_context = np.zeros(64, dtype="float32")
        # Create the queue before the stream so early callbacks have somewhere
        # to put data; the worker is only started once the stream opens cleanly.
        self._vad_queue = queue.Queue(maxsize=32)
        self._vad_worker_thread = None

        def cb(indata, frames, t, status):
            if status:
                log.warning(f"Audio: {status}")
            if not self._recording:
                return
            self._audio_chunks.append(indata.copy())
            self._audio_level = min(1.0, float(np.abs(indata).max()) * 5)
            if self._vad_model is not None:
                try:
                    self._vad_queue.put_nowait(indata.flatten().copy())
                except queue.Full:
                    pass
            # Safety net: a wedged VAD must never record forever. _fire_stop is
            # one-shot and swallows exceptions so it can't abort the PortAudio
            # stream from this realtime callback thread.
            if time.time() - self._record_start >= MAX_RECORD_SECONDS:
                log.warning(f"Max recording length ({MAX_RECORD_SECONDS}s) reached — auto-stopping")
                self._fire_stop()

        # Open the input stream with fallbacks. PortAudio fails in many ways
        # (busy device, unsupported sample rate, "illegal I/O combination",
        # reordered device indices); try several (device, rate) combinations so a
        # transient ALSA quirk does not look like a dead hotkey. Audio is recorded
        # at whatever rate works and resampled to 16kHz in stop_recording().
        try:
            native_rate = SAMPLE_RATE
            try:
                dev_info = (sd.query_devices(device_id) if device_id is not None
                            else sd.query_devices(kind='input'))
                native_rate = int(dev_info["default_samplerate"])
            except Exception as e:
                log.warning(f"Could not query device {device_id} rate: {e}")
            candidates, seen = [], set()
            for dev, rate in [
                (device_id, native_rate), (device_id, SAMPLE_RATE),
                (device_id, 48000), (device_id, 44100),
                (None, native_rate), (None, SAMPLE_RATE),
            ]:
                if rate and (dev, rate) not in seen:
                    seen.add((dev, rate))
                    candidates.append((dev, rate))
            last_err = None
            for dev, rate in candidates:
                stream = None
                try:
                    stream = sd.InputStream(
                        samplerate=rate, channels=1, dtype="float32",
                        blocksize=int(512 * rate / SAMPLE_RATE),
                        device=dev, callback=cb,
                        finished_callback=self._on_stream_finished)
                    self._native_rate = rate  # set before start(): callbacks fire after
                    self._stream = stream
                    stream.start()
                    log.info(f"Recording (dev={dev}, rate={rate})")
                    break
                except Exception as e:
                    last_err = e
                    log.warning(f"Mic open failed (dev={dev}, rate={rate}): {e}")
                    # A stream that opened but failed to start still holds the
                    # ALSA device handle — close it or the next candidate (and
                    # every later recording) can find the device busy.
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                        self._stream = None
            else:
                raise last_err or RuntimeError("No working audio input device")
        except Exception as e:
            log.error(f"Mic failed: {e}")
            with self._lock:
                self._recording = False
            raise

        self.play_start_sound()
        if self._vad_model is not None:
            # Bind the queue + generation into the worker so an orphan from a
            # previous session can never consume the new session's chunks.
            self._vad_worker_thread = threading.Thread(
                target=self._run_vad_worker, args=(self._vad_queue, session_gen),
                daemon=True)
            self._vad_worker_thread.start()
        if self._on_partial is not None and getattr(self.settings, "live_preview", False):
            self._preview_thread = threading.Thread(
                target=self._run_preview_worker, args=(session_gen,), daemon=True)
            self._preview_thread.start()

    def _on_stream_finished(self):
        """PortAudio stream ended. During a normal stop we've already cleared
        _recording; if it's still set, the device died mid-recording (USB mic
        unplugged) — callbacks stop arriving, so VAD and the max-duration cap
        go blind. Fire the stop path so the clip is salvaged."""
        if self._recording:
            log.warning("Input stream ended while recording (device lost?) — auto-stopping")
            self._fire_stop()

    # Preview decodes at most this much trailing audio per tick. Re-transcribing
    # the full buffer every tick was O(n²) in recording length and an in-flight
    # full-clip decode could hold _infer_lock against the final transcription.
    PREVIEW_TAIL_SECONDS = 15

    def _run_preview_worker(self, session_gen):
        """Live preview: periodically transcribe recent audio and emit a partial
        via on_partial. Best-effort — serialized with the final result by
        _infer_lock, and never blocks or alters it. Holds the session generation
        so an orphan (join timed out) exits instead of emitting stale text into
        the next recording."""
        interval = max(0.6, float(getattr(self.settings, "preview_interval", 1.5)))
        last_len = 0

        def current():
            return self._recording and self._session_gen == session_gen

        while current():
            time.sleep(interval)
            if not current():
                break
            chunks = list(self._audio_chunks)
            if not chunks:
                continue
            try:
                audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
                # audio is still at the native capture rate here (resample is
                # below), so gate on native-rate samples, not the 16 kHz const.
                if len(audio) < self._native_rate * 0.6 or len(audio) == last_len:
                    continue
                last_len = len(audio)
                tail = int(self.PREVIEW_TAIL_SECONDS * self._native_rate)
                prefix = "… " if len(audio) > tail else ""
                audio = audio[-tail:]
                if self._native_rate != SAMPLE_RATE:
                    audio = self._resample_linear(audio, self._native_rate, SAMPLE_RATE)
                if self.settings.transcription_backend == BACKEND_REMOTE:
                    text = self._transcribe_remote(audio)
                else:
                    text = self._transcribe_local(audio)
                if text and current() and self._on_partial:
                    self._on_partial(prefix + text)
            except Exception as e:
                log.debug(f"preview worker: {e}")

    def _stop_vad_worker(self):
        """Signal the VAD worker to drain its queue and exit, then join it."""
        if self._vad_queue is not None:
            # put with a short timeout, retried: the old get_nowait/put_nowait
            # dance could lose the sentinel if the worker drained the queue
            # between the two calls, leaving the worker blocked forever.
            for _ in range(3):
                try:
                    self._vad_queue.put(None, timeout=0.25)  # sentinel
                    break
                except queue.Full:
                    continue
        if self._vad_worker_thread is not None:
            self._vad_worker_thread.join(timeout=2.0)
            self._vad_worker_thread = None

    def stop_recording(self):
        with self._lock:
            if not self._recording or self._stop_in_progress:
                return None, 0.0
            self._recording = False
            # Block any new start until this stop fully releases the stream and
            # workers — the joins below take up to ~4s and previously ran
            # unlocked, letting a quick re-press get its fresh session clobbered.
            self._stop_in_progress = True
            stream = self._stream
            self._stream = None
        try:
            # Drain VAD before reading _speech_detected so a late chunk isn't lost.
            self._stop_vad_worker()
            if self._preview_thread is not None:
                self._preview_thread.join(timeout=2.0)  # _infer_lock guards the rest
                self._preview_thread = None
            duration = time.time() - self._record_start
            self.play_stop_sound()
            try:
                if stream:
                    stream.stop()
                    stream.close()
            except Exception as e:
                log.error(f"Stream close: {e}")

            audio = (np.concatenate(self._audio_chunks, axis=0).flatten().astype(np.float32)
                     if self._audio_chunks else np.array([], dtype=np.float32))

            # VAD can miss speech entirely (mis-tuned threshold, worker error, quiet
            # mic). Don't silently drop audio that clearly contains an utterance:
            # if the peak clears the energy floor and the clip is long enough,
            # transcribe anyway and let Whisper's own VAD filter clean it up.
            if not self._speech_detected:
                peak = float(np.abs(audio).max()) if len(audio) else 0.0
                has_energy = peak >= NO_SPEECH_PEAK_FLOOR and len(audio) >= self._native_rate * 0.4
                if not has_energy:
                    log.info(f"No speech (peak={peak:.4f}, {len(audio)} samples)")
                    return None, duration
                log.info(f"VAD missed speech but audio has energy (peak={peak:.4f}) — transcribing anyway")

            # Resample to SAMPLE_RATE using float64 indices. The old float32-index
            # path overflowed past len(audio)-1 on long clips (>~16.7M samples) and
            # crashed with IndexError, which killed the hotkey listener thread.
            if self._native_rate != SAMPLE_RATE and len(audio) > 1:
                audio = self._resample_linear(audio, self._native_rate, SAMPLE_RATE)
                log.info(f"Resampled {self._native_rate}->{SAMPLE_RATE} Hz ({len(audio)} samples)")
            log.info(f"Recorded {duration:.1f}s, {len(audio)} samples")
            return audio, duration
        finally:
            with self._lock:
                self._stop_in_progress = False

    @staticmethod
    def _resample_linear(audio, src_rate, dst_rate):
        """Linear resample with float64 indices (float32 indices overflow on long clips)."""
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if len(audio) <= 1 or src_rate == dst_rate:
            return audio
        num_samples = max(1, int(round(len(audio) * dst_rate / src_rate)))
        x = np.linspace(0.0, len(audio) - 1, num_samples)  # float64 — safe
        return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)

    def _resample_vad_chunk(self, chunk):
        """Resample a raw audio chunk from native device rate to 16kHz for VAD."""
        return self._resample_linear(chunk, self._native_rate, SAMPLE_RATE)

    def _run_vad_worker(self, vad_queue, session_gen):
        """VAD inference loop — runs on a dedicated background thread, not the
        audio callback. Operates on ITS OWN queue + generation: an orphan from a
        previous session must never consume the next session's chunks or fire a
        stale auto-stop into it."""
        while True:
            try:
                chunk = vad_queue.get(timeout=1.0)
            except queue.Empty:
                if self._session_gen != session_gen:
                    break  # orphaned — session moved on without our sentinel
                continue
            if chunk is None:  # sentinel — drain complete, exit cleanly
                break
            if self._session_gen != session_gen:
                break
            try:
                self._check_vad(self._resample_vad_chunk(chunk))
            except Exception as e:
                log.error(f"VAD worker: {e}")

    # ── Transcription ─────────────────────────────────────────────────────
    def transcribe(self, audio, on_segment=None):
        with self._lock:
            self._transcribing = True
        try:
            if len(audio) < SAMPLE_RATE * 0.3:
                return ""
            if self.settings.transcription_backend == BACKEND_REMOTE:
                raw = self._transcribe_remote(audio)
            else:
                raw = self._transcribe_local(audio, on_segment)
            if not raw:
                return ""
            # Dictionary Layer B: deterministic correction on RAW text, before the
            # LLM sees it \u2014 locks canonical spellings even on the raw-fallback path.
            corrected = self.dictionary.apply_replacements(raw)
            if corrected != raw:
                log.debug(f"Dictionary applied: {raw[:60]!r} -> {corrected[:60]!r}")
            log.debug(f"Raw: {corrected[:100]!r}")
            return corrected
        except TranscriptionError:
            raise
        except Exception as e:
            log.error(f"Transcribe: {e}")
            raise TranscriptionError(str(e)) from e
        finally:
            with self._lock:
                self._transcribing = False

    def _lang_and_prompt(self):
        lang = LANGS.get(self.settings.lang_name)
        prompt = ("Transcription en fran\u00e7ais canadien."
                  if self.settings.lang_name == "Fran\u00e7ais (CA)" else None)
        return lang, prompt

    def _transcribe_local(self, audio, on_segment=None):
        with self._lock:
            model = self._whisper_model
        if model is None:
            raise RuntimeError("Whisper model is not loaded")
        lang, prompt = self._lang_and_prompt()
        # Dictionary Layer A: bias the decode toward the user's spellings.
        hotwords = self.dictionary.hotwords_string() or None
        # _infer_lock serializes the model: the live-preview thread and the final
        # transcription must never call faster-whisper concurrently.
        with self._infer_lock:
            segs, _ = model.transcribe(
                audio, language=lang, beam_size=1,
                initial_prompt=prompt, hotwords=hotwords, vad_filter=True)
            parts = []
            for s in segs:
                parts.append(s.text)
                if on_segment:
                    on_segment(" ".join(parts).strip())
            return " ".join(parts).strip()

    def _transcribe_remote(self, audio):
        """POST the clip to the OpenAI-compatible remote ASR server (e.g. P620)."""
        lang, prompt = self._lang_and_prompt()
        fields = {}
        if lang:
            fields["language"] = lang
        if prompt:
            fields["prompt"] = prompt
        hotwords = self.dictionary.hotwords_string()
        if hotwords:
            fields["hotwords"] = hotwords
        # whisper.cpp's whisper-server returns plain text unless asked for
        # JSON; the client always json.loads the body. Both backends accept
        # this field (server.py ignores the unknown part).
        fields["response_format"] = "json"
        wav = _audio_to_wav_bytes(audio, SAMPLE_RATE)
        url = self.settings.remote_url.rstrip("/") + "/audio/transcriptions"
        try:
            with self._infer_lock:  # final POST waits for any in-flight preview POST
                body = _post_multipart_audio(url, wav, fields, timeout=60.0,
                                             token=getattr(self.settings, "remote_token", ""))
            return (body.get("text") or "").strip()
        except Exception as e:
            log.error(f"Remote ASR failed ({self.settings.remote_url}): {e}")
            raise TranscriptionError(
                f"remote ASR unreachable ({self.settings.remote_url})") from e

    # ── AI cleaning ───────────────────────────────────────────────────────
    def clean_text(self, text, preset=None):
        # Raw mode: skip the LLM entirely and keep the exact transcribed words.
        preset = preset or self.settings.cleaning_preset
        if preset == RAW_PRESET:
            return CleanResult(text=self.dictionary.apply_snippets(text))
        try:
            provider = self._get_cleaning_provider(preset)
            if provider is None:
                log.warning("No cleaning provider configured — returning raw")
                return CleanResult(text=self.dictionary.apply_snippets(text),
                                   used_fallback=True, reason="provider unavailable")

            log.debug(f"Cleaning [{self.settings.clean_provider}/{self.settings.clean_model}/{preset}]: {text[:120]!r}")
            word_count = len(text.split())
            max_tokens = max(1024, int(word_count * 2.5))
            result = provider.clean(text, max_tokens=max_tokens)
            # Guard rail: an empty/blank result (safety block, model quirk)
            # must never replace the dictation — regardless of raw length.
            if not result.strip():
                log.warning("Clean returned empty output — using raw")
                return CleanResult(text=self.dictionary.apply_snippets(text),
                                   used_fallback=True, reason="empty cleaned output")
            # Sanity check: cleaned text should be at least 40% of raw length
            if len(result) < len(text) * 0.4 and len(text) > 100:
                log.warning(f"Clean too short ({len(result)} vs {len(text)} raw chars) — using raw")
                return CleanResult(text=self.dictionary.apply_snippets(text),
                                   used_fallback=True, reason="cleaned output too short")
            # Guard rail: small local models sometimes translate FR<->EN
            # wholesale instead of cleaning (seen live with qwen3.5-4b).
            if translation_detected(text, result):
                log.warning("Clean flipped the language (translation) — using raw")
                return CleanResult(text=self.dictionary.apply_snippets(text),
                                   used_fallback=True, reason="cleaner translated the text")
            # Layer D: expand voice snippets AFTER cleaning (LLM never sees the body).
            result = self.dictionary.apply_snippets(result)
            log.debug(f"Clean: {result[:120]!r}")
            return CleanResult(text=result)
        except Exception as e:
            log.error(f"Cleaning API: {e}")
            self._cleaning_provider = None
            return CleanResult(text=self.dictionary.apply_snippets(text),
                               used_fallback=True, reason=str(e))

    # ── VAD ────────────────────────────────────────────────────────────────
    def _vad_thresholds(self):
        """Effective (start, stop) VAD thresholds. Quiet/whisper mode lowers
        both so faint speech still trips speech-start and silence auto-stop."""
        if getattr(self.settings, "quiet_mode", False):
            return (min(self.settings.vad_threshold, 0.30),
                    min(self.settings.vad_neg_thresh, 0.20))
        return self.settings.vad_threshold, self.settings.vad_neg_thresh

    def _check_vad(self, chunk):
        pos_thresh, neg_thresh = self._vad_thresholds()
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        for off in range(0, len(chunk) - 511, 512):
            w = chunk[off:off + 512].astype(np.float32)
            frame = np.concatenate([self._vad_context, w]).reshape(1, -1).astype(np.float32)
            self._vad_context = w[-64:].copy()
            # ONNX is strict about dtype: any double slipping into input/h/c throws
            # "Unexpected input data type", which silently kills speech detection.
            probs, h, c = self._vad_model.session.run(
                None, {"input": frame, "h": self._vad_h, "c": self._vad_c})
            self._vad_h = np.asarray(h, dtype=np.float32)
            self._vad_c = np.asarray(c, dtype=np.float32)
            p = float(probs[0])
            if p >= pos_thresh:
                self._speech_detected = True
                self._silence_start = None
            elif p < neg_thresh and self._speech_detected:
                if self._silence_start is None:
                    self._silence_start = time.time()
                elif time.time() - self._silence_start >= self.settings.silence_timeout:
                    if self._vad_autostop:  # disabled in push-to-talk
                        self._fire_stop()  # one-shot guarded — safe to reach repeatedly
                    return

    # ── Cleaning provider cache ──────────────────────────────────────────
    def _get_cleaning_provider(self, preset_name=None):
        preset_name = preset_name or self.settings.cleaning_preset
        if preset_name == RAW_PRESET:
            return None  # Raw mode never cleans
        # Cache keyed by (provider, model, preset) so per-app preset switching
        # doesn't rebuild the provider every dictation — only when the effective
        # preset actually changes.
        key = (self.settings.clean_provider, self.settings.clean_model, preset_name)
        if self._cleaning_provider is None or self._cleaning_provider_key != key:
            try:
                provider_name = self.settings.clean_provider
                api_key = self.settings.get_api_key().strip()
                needs_key = PROVIDERS.get(provider_name, {}).get("needs_key", True)
                if needs_key and not api_key:
                    return None
                system_prompt = CLEANING_PRESETS.get(preset_name, CLEANING_PRESETS["General"])
                # Dictionary Layer C: tell the LLM these spellings are intentional.
                system_prompt = system_prompt + self.dictionary.prompt_block()
                endpoint = None
                if provider_name == "Ollama":
                    endpoint = self.settings.ollama_endpoint
                elif provider_name == "Local":
                    endpoint = self.settings.local_endpoint
                # A busy local box (shared GPU) needs more headroom than cloud.
                self._cleaning_provider = create_provider(
                    provider_name, api_key, self.settings.clean_model, system_prompt,
                    endpoint=endpoint,
                    timeout=30.0 if provider_name in ("Ollama", "Local") else 15.0)
                self._cleaning_provider_key = key
            except Exception as e:
                log.error(f"Provider init [{self.settings.clean_provider}]: {e}")
                return None
        return self._cleaning_provider

    def invalidate_provider(self):
        self._cleaning_provider = None
        self._cleaning_provider_key = None

    # ── Audio feedback ────────────────────────────────────────────────────
    @staticmethod
    def _play_wav_bytes(wav_data):
        """Play in-memory WAV bytes cross-platform."""
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(wav_data, winsound.SND_MEMORY)
        else:
            import wave
            with wave.open(io.BytesIO(wav_data), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                sd.play(audio, samplerate=wf.getframerate(), blocking=True)

    def play_start_sound(self):
        if self.settings.audio_feedback:
            threading.Thread(
                target=lambda: self._play_wav_bytes(BEEP_START),
                daemon=True).start()

    def play_stop_sound(self):
        if self.settings.audio_feedback:
            threading.Thread(
                target=lambda: self._play_wav_bytes(BEEP_STOP),
                daemon=True).start()

    def get_device_id(self, devices):
        for idx, name in devices:
            if name == self.settings.mic_name:
                return idx
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# OVERLAY — Floating pill indicator (like Lado / Otter)
# ═══════════════════════════════════════════════════════════════════════════════
class FloatingOverlay:
    """Compact floating indicator pill — minimal, always-on-top.

    ZenSEO design language: a dark hairline-bordered pill with a state dot, a
    short label, a live audio-level bar (teal, breathing while recording;
    amber shimmer while transcribing), and a mono timer. State is communicated
    by color + motion, not cluttered text.
    """

    def __init__(self, root):
        self._root = root
        self._win = None
        self._dot = None
        self._bar = None
        self._label = None
        self._timer = None
        self._pulse_job = None
        self._level = 0.0
        self._target_level = 0.0
        self._state = "idle"
        self._monitors_cache = None      # cached xrandr result (rarely changes)
        self._monitors_cache_time = 0.0

    def show(self, state="recording"):
        if self._win is None:
            self._create()
        # Reposition every show — the main window is usually withdrawn, so the
        # old once-per-lifetime position landed on the wrong monitor.
        self._reposition()
        self._state = state
        if state == "recording":
            self._dot.configure(text_color=T.REC)
            self._label.configure(text="Recording", text_color=T.TEXT)
            self._timer.configure(text="0:00")
            self._timer.pack(side="right", padx=(0, 12))
            self._bar.configure(fg_color=T.REC)
            self._start_pulse()
        elif state == "transcribing":
            self._stop_pulse()
            self._dot.configure(text_color=T.AMBER)
            self._label.configure(text="Transcribing", text_color=T.TEXT)
            self._timer.pack_forget()
            self._bar.configure(fg_color=T.AMBER)
            self._start_shimmer()
        self._win.deiconify()
        self._win.lift()

    def hide(self):
        self._stop_pulse()
        if self._win:
            self._win.withdraw()

    def update_timer(self, text):
        if self._timer:
            self._timer.configure(text=text)

    def set_label(self, text):
        if self._label:
            self._label.configure(text=text)

    def set_level(self, level):
        """Feed audio level (0.0–1.0) for the live bar."""
        self._target_level = max(0.0, min(1.0, level))

    def _pointer_monitor(self):
        """Geometry of the monitor under the mouse pointer — where the user is
        actually dictating, unlike the usually-withdrawn main window."""
        px = py = None
        try:
            out = subprocess.run(["xdotool", "getmouselocation", "--shell"],
                                 capture_output=True, text=True, timeout=1)
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    if line.startswith("X="):
                        px = int(line[2:])
                    elif line.startswith("Y="):
                        py = int(line[2:])
        except Exception:
            pass
        # xrandr is the slow part (2s timeout) and monitor layout rarely
        # changes — cache it for 30s so back-to-back dictations don't re-spawn
        # it on the Tk thread. The fast xdotool pointer read above stays live.
        monitors = self._monitors_cache
        if monitors is None or (time.time() - self._monitors_cache_time) > 30:
            monitors = []
            try:
                result = subprocess.run(["xrandr", "--listmonitors"],
                                        capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    import re
                    for line in result.stdout.strip().split("\n")[1:]:
                        m = re.search(r'(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)', line)
                        if m:
                            monitors.append(tuple(int(m.group(i)) for i in (1, 2, 3, 4)))
            except Exception:
                pass
            self._monitors_cache = monitors
            self._monitors_cache_time = time.time()
        if px is not None and py is not None:
            for mw, mh, mx, my in monitors:
                if mx <= px < mx + mw and my <= py < my + mh:
                    return mw, mh, mx, my
        if monitors:
            return monitors[0]
        return self._root.winfo_screenwidth(), self._root.winfo_screenheight(), 0, 0

    def _reposition(self):
        try:
            scale = self._win._get_widget_scaling()
        except Exception:
            scale = detect_ui_scale()
        w, h = int(168 * scale), int(34 * scale)
        mon_w, mon_h, mon_x, mon_y = self._pointer_monitor()
        x = mon_x + (mon_w - w) // 2
        y = mon_y + mon_h - h - 64
        self._win.geometry(f"{w}x{h}+{x}+{y}")

    def _create(self):
        self._win = ctk.CTkToplevel(self._root)
        self._win.overrideredirect(True)
        self._win.attributes('-topmost', True)
        self._win.attributes('-alpha', 0.96)
        self._win.configure(fg_color=T.SIDEBAR)

        pill = ctk.CTkFrame(self._win, fg_color=T.SIDEBAR,
                            corner_radius=T.R_PILL, border_width=1,
                            border_color=T.BORDER)
        pill.pack(fill="both", expand=True, padx=1, pady=1)

        # State dot
        self._dot = ctk.CTkLabel(pill, text="●", font=T.font(11),
                                 text_color=T.REC, width=14)
        self._dot.pack(side="left", padx=(12, 4))

        # Live audio-level bar (teal, grows with level)
        self._bar = ctk.CTkFrame(pill, fg_color=T.REC, corner_radius=2, width=3)
        self._bar.pack(side="left", padx=(2, 8), pady=8)

        # State label
        self._label = ctk.CTkLabel(pill, text="Recording", font=T.font(11, "medium"),
                                   text_color=T.TEXT, anchor="w")
        self._label.pack(side="left")

        # Timer (mono)
        self._timer = ctk.CTkLabel(pill, text="0:00", font=T.font(10, mono=True),
                                   text_color=T.MUTED)
        self._timer.pack(side="right", padx=(0, 12))

        self._win.withdraw()

    # ── Animations ──────────────────────────────────────────────────────
    def _start_pulse(self):
        self._level = 0.3
        self._pulse()

    def _stop_pulse(self):
        if self._pulse_job:
            self._root.after_cancel(self._pulse_job)
            self._pulse_job = None

    def _pulse(self):
        # Smooth toward target level, with a gentle heartbeat baseline
        if self._state != "recording":
            return
        baseline = 0.25
        self._level += (self._target_level - self._level) * 0.4
        display = max(self._level, baseline)
        bar_h = max(4, int(display * 18))
        try:
            self._bar.configure(height=bar_h)
        except Exception:
            pass
        self._pulse_job = self._root.after(60, self._pulse)

    def _start_shimmer(self):
        """Amber bar shimmer for the transcribing state."""
        self._shimmer_on = True
        self._shimmer_phase = 0
        self._shimmer()

    def _shimmer(self):
        if not getattr(self, "_shimmer_on", False) or self._state != "transcribing":
            return
        self._shimmer_phase = (getattr(self, "_shimmer_phase", 0) + 1) % 8
        color = T.AMBER if self._shimmer_phase < 4 else T.AMBER_DIM
        try:
            self._bar.configure(fg_color=color, height=12)
        except Exception:
            pass
        self._pulse_job = self._root.after(120, self._shimmer)


# ═══════════════════════════════════════════════════════════════════════════════
# APP — GUI + Tray + Hotkey + Orchestration
# ═══════════════════════════════════════════════════════════════════════════════
class ZenVoxApp:
    # Palette proxies the shared theme (ZenSEO design language). Kept as class
    # attributes so the many `self.BG` / `self.TEAL` call sites keep working.
    BG     = T.BG
    PANEL  = T.PANEL
    PANEL2 = T.PANEL2
    TEXT   = T.TEXT
    MUTED  = T.MUTED
    FAINT  = T.FAINT
    TEAL   = T.BRAND
    TEAL_H = T.BRAND_HOVER
    BORDER = T.BORDER
    SIDEBAR = T.SIDEBAR

    def __init__(self):
        self.settings = Settings.load()
        self.input_devs = list_input_devices()
        if not self.settings.mic_name and self.input_devs:
            self.settings.mic_name = self.input_devs[0][1]

        self.engine = ZenVoxEngine(self.settings)
        self.history = History()
        self.last_text = ""
        self._last_pasted = ""
        self._timer_job = None
        self._paste_target_window = None
        self._is_first_run = not SETTINGS_FILE.exists()
        self._state_lock = threading.Lock()
        self._clipboard_lock = threading.Lock()
        self._state = AppState.LOADING
        self._state_since = time.time()
        self._stopping = False
        self._load_generation = 0
        self._pending_loads = 0
        self._last_clean_reason = ""
        # Pipeline generation: bumped by the watchdog when a task wedges, so
        # the stale task can never stomp state or paste old text later; also
        # lets the watchdog swap in a fresh executor (the wedged one's single
        # worker would otherwise queue every future stop behind the hang).
        self._pipeline_gen = 0
        self._pipeline_deadline = None  # dynamic stuck-timeout (long clips on CPU)
        self._prepaste_state = None     # clipboard snapshot taken during cleaning
        self._current_view = "home"
        self._shutting_down = False
        self._hotkey_gen = 0            # bumped when a hotkey change respawns the listener
        self._last_latency = None       # (transcribe_ms, clean_ms, paste_ms)
        self._tray_backend = pystray.Icon.__module__
        self._tray_has_menu = pystray.Icon.HAS_MENU
        self._load_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="zenvox-model-loader")
        self._pipeline_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="zenvox-pipeline")

        ctk.set_appearance_mode("dark")
        self._ui_scale = detect_ui_scale()
        ctk.set_widget_scaling(self._ui_scale)
        ctk.set_window_scaling(self._ui_scale)
        self.root = ctk.CTk()
        self.root.title("ZenVox - Voice to Text")
        # Geometry is in LOGICAL pixels — customtkinter multiplies by the window
        # scaling internally. Resizable now (was a fixed 1800x1750 HiDPI hack).
        self.root.geometry("1100x740")
        self.root.minsize(940, 620)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._hide_window)
        self.root.bind("<Control-q>", lambda e: self._quit_from_ui())
        self.root.bind("<Escape>", lambda e: self._hide_window())
        if sys.platform == "win32":
            ico = APP_DIR / "zenvox.ico"
            if ico.exists():
                self.root.iconbitmap(str(ico))
        else:
            png = APP_DIR / "zenvox_logo.png"
            if png.exists():
                from PIL import ImageTk
                self._app_icon = ImageTk.PhotoImage(file=str(png))
                self.root.iconphoto(True, self._app_icon)

        self._build_gui()
        self._sync_provider_inputs()
        self._overlay = FloatingOverlay(self.root)
        self._refresh_history()
        if (sys.platform.startswith("linux")
                and _linux_session_type() == "wayland"
                and not shutil.which("wtype")):
            # pyautogui/xdotool can't inject on Wayland; without wtype the
            # auto-paste silently no-ops and the user gets text with no paste.
            log.warning("Wayland without 'wtype': auto-paste won't work — "
                        "install wtype or use Clipboard-only mode.")
            self.root.after(1500, lambda: self.gui_status.set(
                "Install 'wtype' for auto-paste on Wayland"))

        if self._is_first_run or os.environ.get("ZENVOX_SHOW_WINDOW") == "1":
            # Show the window on first run (to configure) or when explicitly asked.
            self.root.deiconify()
            self.root.lift()
            if self._is_first_run:
                self.settings.save()  # Create settings.json so next launch is normal
        else:
            self.root.withdraw()

        self.icon = pystray.Icon("zenvox", ICONS["loading"],
                                 "ZenVox - Loading...", menu=self._build_menu())

        self._schedule_model_load(self.settings.model_name)
        threading.Thread(target=self._run_hotkey_listener, daemon=True).start()

        if self._tray_backend in ("pystray._appindicator", "pystray._gtk"):
            threading.Thread(target=self.icon.run, daemon=True).start()
        else:
            self.icon.run_detached()
        self.root.after(3000, self._watchdog_tick)
        self.root.mainloop()

    # ── GUI Build (sidebar shell + view router) ───────────────────────────
    NAV_ITEMS = [
        ("home", "Dictate"),
        ("history", "History"),
        ("dictionary", "Dictionary"),
        ("settings", "Settings"),
    ]

    # ── Styled-widget helpers (single source of widget styling) ───────────
    def _combo_kw(self):
        return dict(fg_color=T.PANEL2, border_color=T.BORDER, button_color=T.PANEL2,
                    button_hover_color=T.BRAND, dropdown_fg_color=T.PANEL,
                    dropdown_hover_color=T.BRAND_DIM, dropdown_text_color=T.TEXT,
                    text_color=T.TEXT, font=T.F_LABEL, corner_radius=T.R_CTRL)

    def _entry_kw(self):
        return dict(fg_color=T.PANEL2, border_color=T.BORDER, text_color=T.TEXT,
                    font=T.F_LABEL, corner_radius=T.R_CTRL)

    def _primary_btn(self, parent, text, command, **kw):
        return ctk.CTkButton(parent, text=text, command=command, fg_color=T.BRAND,
                             hover_color=T.BRAND_HOVER, text_color=T.ON_BRAND,
                             font=T.font(12, "semibold"), corner_radius=T.R_CTRL, **kw)

    def _ghost_btn(self, parent, text, command, danger=False, **kw):
        return ctk.CTkButton(parent, text=text, command=command, fg_color=T.PANEL2,
                             hover_color=(T.REC if danger else T.BRAND_DIM),
                             text_color=T.TEXT, font=T.font(12, "medium"),
                             corner_radius=T.R_CTRL, border_width=1, border_color=T.BORDER, **kw)

    def _check(self, parent, variable, text, command=None):
        return ctk.CTkCheckBox(parent, variable=variable, text=text, fg_color=T.BRAND,
                               hover_color=T.BRAND_HOVER, text_color=T.MUTED,
                               font=T.F_SMALL, checkbox_width=18, checkbox_height=18,
                               border_color=T.BORDER, command=command)

    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=T.PANEL, border_color=T.BORDER,
                            border_width=1, corner_radius=T.R_CARD, **kw)

    def _build_gui(self):
        self.root.configure(fg_color=T.BG)
        self.gui_status = ctk.StringVar(value="Loading…")
        self._views = {}
        self._nav_buttons = {}

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ─── Sidebar nav (always-dark, Slack/Discord pattern) ───
        sidebar = ctk.CTkFrame(self.root, fg_color=T.SIDEBAR, corner_radius=0, width=212)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

        # ─── Content area ───
        content = ctk.CTkFrame(self.root, fg_color=T.BG, corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(content, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=32, pady=(24, 10))
        self._view_title = ctk.CTkLabel(header, text="Dictate",
                                        font=T.F_TITLE, text_color=T.TEXT)
        self._view_title.pack(side="left")
        self._status_dot = ctk.CTkLabel(header, text="●", font=T.font(12),
                                        text_color=T.MUTED, width=16)
        self._status_dot.pack(side="right", padx=(8, 0))
        ctk.CTkLabel(header, textvariable=self.gui_status, font=T.F_LABEL,
                     text_color=T.MUTED).pack(side="right")

        container = ctk.CTkFrame(content, fg_color="transparent")
        container.grid(row=1, column=0, sticky="nsew", padx=32, pady=(0, 24))
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        for name, _ in self.NAV_ITEMS:
            frame = ctk.CTkFrame(container, fg_color="transparent")
            frame.grid(row=0, column=0, sticky="nsew")
            self._views[name] = frame

        self._build_view_home(self._views["home"])
        self._build_view_history(self._views["history"])
        self._build_view_dictionary(self._views["dictionary"])
        self._build_view_settings(self._views["settings"])
        self._show_view("home")

    def _build_sidebar(self, parent):
        wm = ctk.CTkFrame(parent, fg_color="transparent")
        wm.pack(fill="x", padx=24, pady=(26, 24))
        ctk.CTkLabel(wm, text="Zen", font=T.font(22, "bold"),
                     text_color=T.TEXT).pack(side="left")
        ctk.CTkLabel(wm, text="Vox", font=T.font(22, "bold"),
                     text_color=T.BRAND).pack(side="left")
        for name, label in self.NAV_ITEMS:
            btn = ctk.CTkButton(
                parent, text=label, anchor="w", fg_color="transparent",
                hover_color=T.PANEL, text_color=T.MUTED, font=T.font(14, "medium"),
                corner_radius=T.R_CTRL, height=40, command=lambda n=name: self._show_view(n))
            btn.pack(fill="x", padx=14, pady=2)
            self._nav_buttons[name] = btn
        ctk.CTkFrame(parent, fg_color="transparent").pack(fill="both", expand=True)
        self._device_label = ctk.CTkLabel(parent, text="", font=T.F_SMALL,
                                           text_color=T.FAINT, anchor="w")
        self._device_label.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkButton(parent, text="Hide to tray", command=self._hide_window,
                      fg_color=T.PANEL, hover_color=T.PANEL2, text_color=T.MUTED,
                      font=T.font(12, "medium"), corner_radius=T.R_CTRL, height=34
                      ).pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkButton(parent, text="Quit", command=self._quit_from_ui,
                      fg_color="transparent", hover_color="#3A1B1A", text_color=T.MUTED,
                      font=T.font(12, "medium"), corner_radius=T.R_CTRL, height=34,
                      border_width=1, border_color=T.BORDER
                      ).pack(fill="x", padx=14, pady=(0, 18))

    def _show_view(self, name):
        view = self._views.get(name)
        if view is None:
            return
        self._current_view = name
        view.tkraise()
        self._view_title.configure(text=dict(self.NAV_ITEMS).get(name, name))
        for n, btn in self._nav_buttons.items():
            active = (n == name)
            btn.configure(text_color=T.BRAND if active else T.MUTED,
                          fg_color=T.BRAND_DIM if active else "transparent")
        if name == "dictionary":
            self._refresh_dictionary()
        elif name == "history":
            self._refresh_history()
        elif name == "home":
            self._refresh_stats()

    # ── View: Dictate (home) ──────────────────────────────────────────────
    def _build_view_home(self, parent):
        # Stat band — words / dictations / avg WPM / streak (mono numerals)
        band = ctk.CTkFrame(parent, fg_color="transparent")
        band.pack(fill="x", pady=(0, 14))
        self._stat_vars = {}
        stat_defs = [("today_words", "words today"), ("today_count", "dictations today"),
                     ("wpm", "avg WPM"), ("streak", "day streak")]
        for i, (key, label) in enumerate(stat_defs):
            band.grid_columnconfigure(i, weight=1, uniform="stat")
            cell = self._card(band)
            cell.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            inner = ctk.CTkFrame(cell, fg_color="transparent")
            inner.pack(fill="both", expand=True, padx=16, pady=(14, 12))
            var = ctk.StringVar(value="—")
            self._stat_vars[key] = var
            ctk.CTkLabel(inner, textvariable=var, font=T.F_METRIC,
                         text_color=T.TEXT, anchor="w").pack(fill="x")
            ctk.CTkLabel(inner, text=label, font=T.F_SMALL,
                         text_color=T.MUTED, anchor="w").pack(fill="x", pady=(2, 0))

        # Spotlight record card
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 14))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=20)
        self._rec_btn = ctk.CTkButton(
            inner, text="●", width=58, height=58, corner_radius=29,
            fg_color=T.REC, hover_color=T.REC_HOVER, text_color="#ffffff",
            font=T.font(24), command=self._gui_toggle_record)
        self._rec_btn.pack(side="left", padx=(0, 20))
        meta = ctk.CTkFrame(inner, fg_color="transparent")
        meta.pack(side="left", fill="x", expand=True)
        self._timer_label = ctk.CTkLabel(meta, text="Ready", anchor="w",
                                         font=T.font(15, "semibold"), text_color=T.TEXT)
        self._timer_label.pack(fill="x")
        self._level_bar = ctk.CTkProgressBar(meta, progress_color=T.REC,
                                            fg_color=T.PANEL2, height=8, corner_radius=4)
        self._level_bar.pack(fill="x", pady=(9, 0))
        self._level_bar.set(0)
        hk = self.settings.hotkey_record.lower()
        self._hint_label = ctk.CTkLabel(
            meta, text=f"Press  {hk}  to dictate into any app",
            font=T.F_SMALL, text_color=T.MUTED, anchor="w")
        self._hint_label.pack(fill="x", pady=(7, 0))

        # Last transcription — editable, with correction capture
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(fill="x", pady=(2, 6))
        ctk.CTkLabel(hdr, text="Last transcription", font=T.F_SECTION,
                     text_color=T.MUTED).pack(side="left")
        self._meta_label = ctk.CTkLabel(hdr, text="", font=T.F_MONO_S, text_color=T.FAINT)
        self._meta_label.pack(side="right")
        self.gui_text = ctk.CTkTextbox(parent, wrap="word",
                                       fg_color=T.PANEL, text_color=T.TEXT, font=T.font(14),
                                       border_color=T.BORDER, border_width=1,
                                       corner_radius=T.R_CARD)
        self.gui_text.pack(fill="both", expand=True)
        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.pack(fill="x", pady=(12, 0))
        # Correction capture (dictionary auto-learning): if the user edits the
        # box, "Save correction" diffs it against the transcript and offers to
        # add proper nouns to the dictionary.
        self._save_correction_btn = self._ghost_btn(
            bf, "Save correction", self._gui_save_correction, width=140, height=38)
        self._save_correction_btn.pack(side="left")
        self._primary_btn(bf, "Copy", self._gui_copy, width=104, height=38).pack(side="right")
        self._ghost_btn(bf, "Re-paste", self._gui_repaste, width=110, height=38
                        ).pack(side="right", padx=(0, 10))

    # ── View: History ─────────────────────────────────────────────────────
    def _build_view_history(self, parent):
        self._search_var = ctk.StringVar()
        se = ctk.CTkEntry(parent, textvariable=self._search_var,
                          placeholder_text="Search transcriptions…", height=40,
                          **self._entry_kw())
        se.pack(fill="x", pady=(0, 12))
        self._search_job = None
        def _debounced(*a):
            if self._search_job is not None:
                try:
                    self.root.after_cancel(self._search_job)
                except Exception:
                    pass
            self._search_job = self.root.after(200, self._refresh_history)
        self._search_var.trace_add("write", _debounced)
        self._hist_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._hist_frame.pack(fill="both", expand=True)
        hbf = ctk.CTkFrame(parent, fg_color="transparent")
        hbf.pack(fill="x", pady=(12, 0))
        self._ghost_btn(hbf, "Clear history", self._gui_clear_history,
                        danger=True, width=126, height=34).pack(side="right")

    # ── View: Dictionary ──────────────────────────────────────────────────
    def _build_view_dictionary(self, parent):
        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(head, anchor="w", font=T.F_LABEL, text_color=T.MUTED, justify="left",
                     text="Words and names come out spelled right every time. "
                          "Snippets expand a spoken trigger into longer text."
                     ).pack(side="left", fill="x", expand=True)
        self._ghost_btn(head, "Import", self._dict_import, width=84, height=30).pack(side="right", padx=(6, 0))
        self._ghost_btn(head, "Export", self._dict_export, width=84, height=30).pack(side="right")

        addcard = self._card(parent)
        addcard.pack(fill="x", pady=(0, 12))
        row = ctk.CTkFrame(addcard, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(14, 8))
        self.gui_dict_kind = ctk.StringVar(value="Term")
        ctk.CTkComboBox(row, variable=self.gui_dict_kind, values=["Term", "Snippet"],
                        width=110, command=lambda v=None: self._on_dict_kind(), **self._combo_kw()
                        ).pack(side="left", padx=(0, 8))
        self._dict_written = ctk.CTkEntry(row, placeholder_text="Written (e.g. ZenVox)",
                                          width=200, **self._entry_kw())
        self._dict_written.pack(side="left", padx=(0, 8))
        self._dict_spoken = ctk.CTkEntry(row, placeholder_text="Sounds like (comma-separated)",
                                         **self._entry_kw())
        self._dict_spoken.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._primary_btn(row, "Add", self._dict_add, width=68, height=32).pack(side="left")

        opts = ctk.CTkFrame(addcard, fg_color="transparent")
        opts.pack(fill="x", padx=16, pady=(0, 14))
        self._dict_boost = ctk.BooleanVar(value=False)
        self._dict_boost_cb = self._check(opts, self._dict_boost, "Boost only (bias, no replace)")
        self._dict_boost_cb.pack(side="left", padx=(0, 16))
        self._dict_case = ctk.BooleanVar(value=False)
        self._dict_case_cb = self._check(opts, self._dict_case, "Case-sensitive")
        self._dict_case_cb.pack(side="left")
        self._dict_written.bind("<Return>", lambda e: self._dict_add())
        self._dict_spoken.bind("<Return>", lambda e: self._dict_add())

        self._dict_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._dict_frame.pack(fill="both", expand=True)

    def _on_dict_kind(self):
        # Snippet body is free text, not "boost only" vocab — hide the toggle.
        is_snippet = self.gui_dict_kind.get() == "Snippet"
        self._dict_written.configure(
            placeholder_text="Expansion text" if is_snippet else "Written (e.g. ZenVox)")
        self._dict_spoken.configure(
            placeholder_text="Trigger phrase(s)" if is_snippet else "Sounds like (comma-separated)")
        if is_snippet:
            self._dict_boost.set(False)
            self._dict_boost_cb.configure(state="disabled")
        else:
            self._dict_boost_cb.configure(state="normal")

    def _refresh_dictionary(self):
        if not hasattr(self, "_dict_frame"):
            return
        for w in self._dict_frame.winfo_children():
            w.destroy()
        entries = sorted(self.engine.dictionary.entries,
                         key=lambda e: (e.is_snippet, e.written.lower()))
        if not entries:
            empty = ctk.CTkFrame(self._dict_frame, fg_color="transparent")
            empty.pack(fill="x", pady=40)
            ctk.CTkLabel(empty, text="No words yet", font=T.font(14, "semibold"),
                         text_color=T.MUTED).pack()
            ctk.CTkLabel(empty, text="Add names, jargon, or snippets above.",
                         font=T.F_SMALL, text_color=T.FAINT).pack(pady=(4, 0))
            return
        for e in entries:
            self._dict_row(e)

    def _dict_row(self, e):
        f = ctk.CTkFrame(self._dict_frame, fg_color=T.PANEL, corner_radius=T.R_CTRL)
        f.pack(fill="x", pady=3)
        inner = ctk.CTkFrame(f, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=9)
        dim = not e.enabled
        ctk.CTkLabel(inner, text=e.written, font=T.font(13, "semibold"),
                     text_color=(T.FAINT if dim else T.BRAND)).pack(side="left")
        badge = "snippet" if e.is_snippet else ("boost" if e.boost_only else None)
        if badge:
            ctk.CTkLabel(inner, text=f"  {badge}", font=T.F_TINY, text_color=T.FAINT).pack(side="left")
        if e.case_sensitive:
            ctk.CTkLabel(inner, text="  Aa", font=T.F_TINY, text_color=T.FAINT).pack(side="left")
        if e.spoken:
            arrow = "→" if e.is_snippet else "←"
            ctk.CTkLabel(inner, text=f"   {arrow} " + ", ".join(e.spoken),
                         font=T.F_SMALL, text_color=(T.FAINT if dim else T.MUTED),
                         anchor="w").pack(side="left", fill="x", expand=True)
        ctk.CTkButton(inner, text="✕", width=28, height=24, fg_color=T.PANEL2,
                      hover_color=T.REC, text_color=T.MUTED, font=T.font(12),
                      corner_radius=T.R_CHIP,
                      command=lambda w=e.written: self._dict_delete(w)).pack(side="right")
        toggle = "On" if e.enabled else "Off"
        ctk.CTkButton(inner, text=toggle, width=42, height=24,
                      fg_color=(T.BRAND_DIM if e.enabled else T.PANEL2),
                      hover_color=T.BORDER, text_color=(T.BRAND if e.enabled else T.MUTED),
                      font=T.F_TINY, corner_radius=T.R_CHIP,
                      command=lambda w=e.written: self._dict_toggle(w)).pack(side="right", padx=(0, 6))

    def _dict_add(self):
        written = self._dict_written.get().strip()
        if not written:
            return
        spoken = [s.strip() for s in self._dict_spoken.get().split(",") if s.strip()]
        kind = "snippet" if self.gui_dict_kind.get() == "Snippet" else "term"
        self.engine.dictionary.add(DictionaryEntry(
            written=written, spoken=spoken, kind=kind,
            boost_only=bool(self._dict_boost.get()),
            case_sensitive=bool(self._dict_case.get())))
        self.engine.dictionary.invalidate_cache()
        self.engine.invalidate_provider()  # Layer C must pick up new vocab
        self._dict_written.delete(0, "end")
        self._dict_spoken.delete(0, "end")
        self._dict_boost.set(False)
        self._dict_case.set(False)
        self._refresh_dictionary()

    def _dict_delete(self, written):
        self.engine.dictionary.delete(written)
        self.engine.dictionary.invalidate_cache()
        self.engine.invalidate_provider()
        self._refresh_dictionary()

    def _dict_toggle(self, written):
        for e in self.engine.dictionary.entries:
            if e.written == written:
                e.enabled = not e.enabled
                break
        self.engine.dictionary.save()
        self.engine.dictionary.invalidate_cache()
        self.engine.invalidate_provider()
        self._refresh_dictionary()

    def _dict_import(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(
            title="Import dictionary", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            n = self.engine.dictionary.import_json(path)
            self.engine.dictionary.invalidate_cache()
            self.engine.invalidate_provider()
            self._refresh_dictionary()
            messagebox.showinfo("Import", f"Imported {n} entries.")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def _dict_export(self):
        from tkinter import filedialog, messagebox
        path = filedialog.asksaveasfilename(
            title="Export dictionary", defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.engine.dictionary.export_json(path)
            messagebox.showinfo("Export", "Dictionary exported.")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ── View: Settings ────────────────────────────────────────────────────
    def _build_view_settings(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        combo = self._combo_kw()
        entry = self._entry_kw()

        def section(title, desc=None):
            ctk.CTkLabel(scroll, text=title, font=T.F_SECTION,
                         text_color=T.TEXT, anchor="w").pack(fill="x", pady=(16, 2))
            if desc:
                ctk.CTkLabel(scroll, text=desc, font=T.F_SMALL, text_color=T.FAINT,
                             anchor="w", justify="left").pack(fill="x", pady=(0, 6))
            else:
                ctk.CTkFrame(scroll, fg_color="transparent", height=4).pack(fill="x")
            card = self._card(scroll)
            card.pack(fill="x")
            return card

        def label(parent, text):
            return ctk.CTkLabel(parent, text=text, font=T.F_LABEL, text_color=T.MUTED)

        # Transcription
        c = section("Transcription")
        r = ctk.CTkFrame(c, fg_color="transparent")
        r.pack(fill="x", padx=16, pady=14)
        self.gui_model = ctk.StringVar(value=self.settings.model_name)
        ctk.CTkComboBox(r, variable=self.gui_model, values=MODELS, width=170,
                        command=self._on_model, **combo).pack(side="left", padx=(0, 8))
        self.gui_lang = ctk.StringVar(value=self.settings.lang_name)
        ctk.CTkComboBox(r, variable=self.gui_lang, values=list(LANGS.keys()), width=150,
                        command=self._on_lang, **combo).pack(side="left", padx=(0, 8))
        self.gui_mic = ctk.StringVar(value=self.settings.mic_name)
        ctk.CTkComboBox(r, variable=self.gui_mic, values=[n for _, n in self.input_devs],
                        command=self._on_mic, **combo).pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(c, anchor="w", font=T.F_SMALL, text_color=T.FAINT,
                     text="Keep Language on Auto-detect for franglais — forcing French mistranscribes English."
                     ).pack(fill="x", padx=16, pady=(0, 8))
        rb = ctk.CTkFrame(c, fg_color="transparent")
        rb.pack(fill="x", padx=16, pady=(0, 14))
        label(rb, "Run on").pack(side="left", padx=(0, 6))
        self._backend_labels = {BACKEND_LOCAL: "This machine (local GPU/CPU)",
                                BACKEND_REMOTE: "Remote server"}
        self.gui_backend = ctk.StringVar(
            value=self._backend_labels.get(self.settings.transcription_backend, self._backend_labels[BACKEND_LOCAL]))
        ctk.CTkComboBox(rb, variable=self.gui_backend, values=list(self._backend_labels.values()),
                        width=220, command=self._on_backend, **combo).pack(side="left", padx=(0, 8))
        self.gui_remote_url = ctk.StringVar(value=self.settings.remote_url)
        rue = ctk.CTkEntry(rb, textvariable=self.gui_remote_url, placeholder_text="http://host:8771/v1", **entry)
        rue.pack(side="left", fill="x", expand=True, padx=(0, 8))
        rue.bind("<FocusOut>", lambda e: self._on_remote_url())
        self._ghost_btn(rb, "Test", self._on_test_remote, width=60, height=28).pack(side="left")
        rt = ctk.CTkFrame(c, fg_color="transparent")
        rt.pack(fill="x", padx=16, pady=(0, 14))
        label(rt, "Remote token").pack(side="left", padx=(0, 6))
        self.gui_remote_token = ctk.StringVar(value=self.settings.remote_token)
        rte = ctk.CTkEntry(rt, textvariable=self.gui_remote_token, show="*",
                           placeholder_text="optional (ZENVOX_ASR_TOKEN)", **entry)
        rte.pack(side="left", fill="x", expand=True)
        rte.bind("<FocusOut>", lambda e: self._on_remote_token())

        # AI Cleaning — provider + key + model MUST share one frame
        c = section("AI Cleaning")
        r = ctk.CTkFrame(c, fg_color="transparent")
        r.pack(fill="x", padx=16, pady=14)
        self.gui_provider = ctk.StringVar(value=self.settings.clean_provider)
        ctk.CTkComboBox(r, variable=self.gui_provider, values=PROVIDER_NAMES, width=120,
                        command=self._on_provider, **combo).pack(side="left", padx=(0, 8))
        self.gui_key = ctk.StringVar(value=self.settings.get_api_key())
        self._key_entry = ctk.CTkEntry(r, textvariable=self.gui_key, show="*",
                                       placeholder_text="API key", width=220, **entry)
        self._key_entry.pack(side="left", padx=(0, 8))
        self._key_entry.bind("<FocusOut>", lambda e: self._on_key())
        self.gui_clean = ctk.StringVar(value=self.settings.clean_model)
        ce = ctk.CTkEntry(r, textvariable=self.gui_clean,
                          placeholder_text=PROVIDERS.get(self.settings.clean_provider, {}).get("default_model", ""),
                          **entry)
        ce.pack(side="left", fill="x", expand=True)
        ce.bind("<FocusOut>", lambda e: self._on_clean())
        self._model_entry = ce
        r2 = ctk.CTkFrame(c, fg_color="transparent")
        r2.pack(fill="x", padx=16, pady=(0, 14))
        label(r2, "Cleaning style").pack(side="left", padx=(0, 8))
        self.gui_preset = ctk.StringVar(value=self.settings.cleaning_preset)
        ctk.CTkComboBox(r2, variable=self.gui_preset, values=PRESET_NAMES,
                        width=150, command=self._on_preset, **combo).pack(side="left")
        ctk.CTkLabel(r2, text="Raw = no AI cleanup, exact words", font=T.F_SMALL,
                     text_color=T.FAINT).pack(side="left", padx=(10, 0))

        # Per-app rules
        c = section("Per-app cleaning rules",
                    "Pick the cleaning style automatically by the app you dictate into "
                    "(Slack sounds casual, Gmail sounds professional). X11 only.")
        self._rules_frame = ctk.CTkFrame(c, fg_color="transparent")
        self._rules_frame.pack(fill="x", padx=16, pady=(12, 8))
        ar = ctk.CTkFrame(c, fg_color="transparent")
        ar.pack(fill="x", padx=16, pady=(0, 14))
        self._rule_pattern = ctk.CTkEntry(ar, placeholder_text="window match (e.g. *slack*|*discord*)",
                                          **entry)
        self._rule_pattern.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.gui_rule_preset = ctk.StringVar(value="General")
        ctk.CTkComboBox(ar, variable=self.gui_rule_preset, values=PRESET_NAMES,
                        width=130, **combo).pack(side="left", padx=(0, 8))
        self._ghost_btn(ar, "This window", self._on_rule_use_window, width=104, height=32).pack(side="left", padx=(0, 8))
        self._primary_btn(ar, "Add", self._on_rule_add, width=64, height=32).pack(side="left")

        # Behavior
        c = section("Behavior")
        r = ctk.CTkFrame(c, fg_color="transparent")
        r.pack(fill="x", padx=16, pady=14)
        label(r, "Silence stop").pack(side="left", padx=(0, 8))
        self.gui_silence = ctk.DoubleVar(value=self.settings.silence_timeout)
        self._silence_val = ctk.CTkLabel(r, text=f"{self.settings.silence_timeout:.1f}s",
                                         font=T.F_MONO_S, text_color=T.TEXT, width=44)
        self._silence_val.pack(side="left", padx=(0, 8))
        ctk.CTkSlider(r, from_=0.5, to=4.0, number_of_steps=35, variable=self.gui_silence,
                      command=self._on_silence, button_color=T.BRAND, button_hover_color=T.BRAND_HOVER,
                      progress_color=T.BRAND, fg_color=T.PANEL2, width=160).pack(side="left", padx=(0, 16))
        label(r, "Output").pack(side="left", padx=(0, 6))
        self.gui_output = ctk.StringVar(value=self.settings.output_mode)
        ctk.CTkComboBox(r, variable=self.gui_output, values=OUTPUT_MODES, width=150,
                        command=self._on_output, **combo).pack(side="left")
        rof = ctk.CTkFrame(c, fg_color="transparent")
        rof.pack(fill="x", padx=16, pady=(0, 4))
        self._output_file_label = ctk.CTkLabel(rof, text="", font=T.F_SMALL,
                                               text_color=T.FAINT, anchor="w")
        self._output_file_label.pack(side="left", fill="x", expand=True)
        self._output_file_btn = self._ghost_btn(rof, "Change file…", self._on_pick_output_file,
                                                width=110, height=28)
        rb = ctk.CTkFrame(c, fg_color="transparent")
        rb.pack(fill="x", padx=16, pady=(6, 14))
        label(rb, "Capture").pack(side="left", padx=(0, 6))
        self._capture_labels = {CAPTURE_TOGGLE: "Toggle — auto-stop on silence",
                                CAPTURE_PTT: "Push-to-talk — hold to talk"}
        self.gui_capture = ctk.StringVar(
            value=self._capture_labels.get(self.settings.capture_mode, self._capture_labels[CAPTURE_TOGGLE]))
        ctk.CTkComboBox(rb, variable=self.gui_capture, values=list(self._capture_labels.values()),
                        width=250, command=self._on_capture_mode, **combo).pack(side="left")
        rc = ctk.CTkFrame(c, fg_color="transparent")
        rc.pack(fill="x", padx=16, pady=(0, 14))
        self.gui_audio = ctk.BooleanVar(value=self.settings.audio_feedback)
        self._check(rc, self.gui_audio, "Sound feedback", self._on_audio).pack(side="left", padx=(0, 18))
        self.gui_preview = ctk.BooleanVar(value=self.settings.live_preview)
        self._check(rc, self.gui_preview, "Live preview", self._on_preview).pack(side="left", padx=(0, 18))
        self.gui_quiet = ctk.BooleanVar(value=self.settings.quiet_mode)
        self._check(rc, self.gui_quiet, "Quiet / whisper mode", self._on_quiet).pack(side="left")

        # Hotkeys — press-to-capture
        c = section("Hotkeys", "Click a field, then press the key combo you want.")
        hk = ctk.CTkFrame(c, fg_color="transparent")
        hk.pack(fill="x", padx=16, pady=14)
        label(hk, "Record / stop").pack(side="left", padx=(0, 8))
        self.gui_hotkey_record = ctk.StringVar(value=self.settings.hotkey_record)
        self._hotkey_rec_btn = ctk.CTkButton(
            hk, textvariable=self.gui_hotkey_record, width=150, height=32,
            fg_color=T.PANEL2, hover_color=T.BRAND_DIM, text_color=T.TEXT,
            font=T.F_MONO_S, corner_radius=T.R_CTRL, border_width=1, border_color=T.BORDER,
            command=lambda: self._capture_hotkey("record"))
        self._hotkey_rec_btn.pack(side="left", padx=(0, 20))
        label(hk, "Re-paste last").pack(side="left", padx=(0, 8))
        self.gui_hotkey_repaste = ctk.StringVar(value=self.settings.hotkey_repaste)
        self._hotkey_rep_btn = ctk.CTkButton(
            hk, textvariable=self.gui_hotkey_repaste, width=150, height=32,
            fg_color=T.PANEL2, hover_color=T.BRAND_DIM, text_color=T.TEXT,
            font=T.F_MONO_S, corner_radius=T.R_CTRL, border_width=1, border_color=T.BORDER,
            command=lambda: self._capture_hotkey("repaste"))
        self._hotkey_rep_btn.pack(side="left")
        if not self._tray_has_menu:
            ctk.CTkLabel(c, anchor="w", font=T.F_SMALL, text_color=T.FAINT,
                         text="Tray menus are unavailable on this backend — use this window; Ctrl+Q quits."
                         ).pack(fill="x", padx=16, pady=(0, 12))

        self._refresh_rules()
        self._sync_output_file_row()

    # ── GUI Callbacks ─────────────────────────────────────────────────────
    def _gui_toggle_record(self):
        """Record button click — same as hotkey toggle. Runs off the Tk thread:
        opening the mic can block for hundreds of ms (unplugged/busy device)
        and would freeze the whole window."""
        threading.Thread(target=self._safe_toggle, daemon=True).start()

    def _copy_to_clipboard(self, text):
        """Copy off the Tk thread, holding _clipboard_lock so a Copy during the
        auto-paste restore window isn't silently reverted by the restore thread."""
        if not text:
            return
        def _do():
            with self._clipboard_lock:
                _copy_text(text)
                self._last_pasted = text
        threading.Thread(target=_do, daemon=True).start()

    def _gui_copy(self):
        self._copy_to_clipboard(self.last_text)

    def _gui_repaste(self):
        if self.last_text:
            threading.Thread(target=self._repaste, daemon=True).start()

    def _gui_update_text(self, text):
        self.gui_text.delete("1.0", "end")
        self.gui_text.insert("end", text)
        # Remember what WE wrote, so "Save correction" can diff against it.
        self._displayed_text = text

    def _gui_save_correction(self):
        """Correction capture (dictionary auto-learning): diff the edited box
        against the last transcription; offer to add new proper nouns and
        re-save the corrected text as last_text."""
        from tkinter import messagebox
        edited = self.gui_text.get("1.0", "end").strip()
        original = getattr(self, "_displayed_text", "")
        if not edited or edited == original:
            self.gui_status.set("No changes to save")
            return
        self.last_text = edited
        self._displayed_text = edited
        # Proper-noun candidates: capitalized/unusual tokens that appear in the
        # edit but not the original (the words the user fixed).
        import re as _re
        orig_words = set(w.lower() for w in _re.findall(r"[\w'-]+", original))
        candidates = []
        seen = set()
        for w in _re.findall(r"[\w'-]+", edited):
            if (len(w) >= 3 and w[0].isupper() and w.lower() not in orig_words
                    and w.lower() not in seen):
                seen.add(w.lower())
                existing = {e.written.lower() for e in self.engine.dictionary.entries}
                if w.lower() not in existing:
                    candidates.append(w)
        added = 0
        for w in candidates[:8]:
            if messagebox.askyesno("Add to dictionary?",
                                   f"Add “{w}” to your dictionary so it's spelled "
                                   f"this way every time?"):
                self.engine.dictionary.add(DictionaryEntry(written=w))
                added += 1
        if added:
            self.engine.dictionary.invalidate_cache()
            self.engine.invalidate_provider()
            self._refresh_dictionary()
        self.gui_status.set(f"Correction saved{f' · {added} added to dictionary' if added else ''}")

    def _refresh_stats(self):
        if not hasattr(self, "_stat_vars"):
            return
        try:
            s = self.history.stats()
        except Exception:
            return
        fmt = {
            "today_words": str(s["today_words"]),
            "today_count": str(s["today_count"]),
            "wpm": str(s["wpm"]) if s["wpm"] else "—",
            "streak": str(s["streak"]),
        }
        for key, var in self._stat_vars.items():
            var.set(fmt.get(key, "—"))

    def _on_partial_preview(self, text):
        # Live preview while still recording — show partial text + a cue.
        if self._get_state() == AppState.RECORDING and text:
            self._gui_update_text(text + " …")
            self._overlay.set_label("Listening…")

    def _gui_clear_history(self):
        from tkinter import messagebox
        try:
            count = self.history.stats().get("total_count", 0)
        except Exception:
            count = 0
        if count and not messagebox.askyesno(
                "Clear history",
                f"Permanently delete all {count} transcription(s)? This cannot be undone."):
            return
        self.history.clear()
        self._refresh_history()
        self._refresh_stats()

    @staticmethod
    def _day_label(dt):
        from datetime import date, timedelta
        today = date.today()
        d = dt.date()
        if d == today:
            return "Today"
        if d == today - timedelta(days=1):
            return "Yesterday"
        # Build the day without %-d (glibc-only; crashes on Windows CPython).
        md = f"{dt.strftime('%b')} {dt.day}"
        return md if d.year == today.year else f"{md}, {dt.year}"

    def _refresh_history(self):
        for w in self._hist_frame.winfo_children():
            w.destroy()
        q = self._search_var.get().strip()
        entries = self.history.search(q) if q else self.history.get_recent(80)
        if not entries:
            empty = ctk.CTkFrame(self._hist_frame, fg_color="transparent")
            empty.pack(fill="x", pady=48)
            msg = f'No matches for “{q}”' if q else "No transcriptions yet"
            ctk.CTkLabel(empty, text=msg, font=T.font(14, "semibold"),
                         text_color=T.MUTED).pack()
            if not q:
                ctk.CTkLabel(empty, text=f"Press {self.settings.hotkey_record.lower()} and start talking.",
                             font=T.F_SMALL, text_color=T.FAINT).pack(pady=(4, 0))
            return
        current_day = None
        for entry in entries:
            try:
                dt = datetime.fromisoformat(entry.timestamp).astimezone()
                ts = dt.strftime("%H:%M")
                day = self._day_label(dt)
            except Exception:
                ts, day = "—", "Earlier"
            if day != current_day:
                current_day = day
                ctk.CTkLabel(self._hist_frame, text=day, font=T.font(11, "semibold"),
                             text_color=T.FAINT, anchor="w").pack(fill="x", padx=4, pady=(12, 4))
            self._history_row(entry, ts)

    def _history_row(self, entry, ts):
        preview = entry.cleaned_text.replace("\n", " ")
        if len(preview) > 88:
            preview = preview[:88] + "…"
        dur = f"{entry.duration_sec:.0f}s" if entry.duration_sec else ""
        card = ctk.CTkFrame(self._hist_frame, fg_color=T.PANEL, corner_radius=T.R_CTRL)
        card.pack(fill="x", pady=2)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(head, text=ts, font=T.F_MONO_S, text_color=T.BRAND,
                     width=40).pack(side="left", padx=(0, 10))
        preview_lbl = ctk.CTkLabel(head, text=preview, font=T.font(12),
                                   text_color=T.TEXT, anchor="w", justify="left")
        preview_lbl.pack(side="left", fill="x", expand=True)
        if entry.cleaning_preset and entry.cleaning_preset != "General":
            ctk.CTkLabel(head, text=entry.cleaning_preset, font=T.F_TINY,
                         text_color=T.FAINT).pack(side="right", padx=(6, 6))
        if dur:
            ctk.CTkLabel(head, text=dur, font=T.F_MONO_S, text_color=T.MUTED,
                         width=34).pack(side="right")

        body = {"open": False, "frame": None}

        def toggle(_=None):
            if body["open"]:
                if body["frame"] is not None:
                    body["frame"].destroy()
                    body["frame"] = None
                body["open"] = False
                return
            fr = ctk.CTkFrame(card, fg_color="transparent")
            fr.pack(fill="x", padx=12, pady=(0, 10))
            box = ctk.CTkTextbox(fr, wrap="word", height=90, fg_color=T.PANEL2,
                                 text_color=T.TEXT, font=T.font(12),
                                 border_color=T.BORDER, border_width=1, corner_radius=T.R_CTRL)
            box.pack(fill="x")
            box.insert("end", entry.cleaned_text)
            box.configure(state="disabled")
            btns = ctk.CTkFrame(fr, fg_color="transparent")
            btns.pack(fill="x", pady=(6, 0))
            ctk.CTkButton(btns, text="Delete", width=68, height=26, fg_color=T.PANEL2,
                          hover_color=T.REC, text_color=T.MUTED, font=T.F_TINY,
                          corner_radius=T.R_CHIP, border_width=1, border_color=T.BORDER,
                          command=lambda: self._history_delete(entry.id)).pack(side="right", padx=(6, 0))
            ctk.CTkButton(btns, text="Copy", width=68, height=26, fg_color=T.BRAND_DIM,
                          hover_color=T.BRAND, text_color=T.BRAND, font=T.F_TINY,
                          corner_radius=T.R_CHIP,
                          command=lambda t=entry.cleaned_text: self._copy_to_clipboard(t)).pack(side="right")
            body["frame"] = fr
            body["open"] = True

        for w in (card, head, preview_lbl):
            w.bind("<Button-1>", toggle)

    def _history_delete(self, entry_id):
        self.history.delete(entry_id)
        self._refresh_history()

    # ── App State ────────────────────────────────────────────────────────
    def _queue_ui(self, callback):
        try:
            self.root.after(0, callback)
        except RuntimeError:
            pass

    def _set_tray(self, icon_img, title):
        """Mutate the tray icon/title on the GTK loop thread.

        pystray's AppIndicator/GTK backends run their own GLib main loop; GTK
        is not thread-safe, so touching self.icon.icon/title from the pipeline
        or hotkey threads can segfault. Route through GLib.idle_add when on a
        GTK backend; fall back to a direct set otherwise.
        """
        def apply():
            self.icon.icon = icon_img
            self.icon.title = title
            return False
        if self._tray_backend in ("pystray._appindicator", "pystray._gtk"):
            try:
                from gi.repository import GLib
                GLib.idle_add(apply)
                return
            except Exception:
                pass
        apply()

    def _rebuild_menu(self):
        """Rebuild + swap the tray menu on the GTK loop thread (see _set_tray)."""
        menu = self._build_menu()
        def apply():
            self.icon.menu = menu
            return False
        if self._tray_backend in ("pystray._appindicator", "pystray._gtk"):
            try:
                from gi.repository import GLib
                GLib.idle_add(apply)
                return
            except Exception:
                pass
        apply()

    def _get_state(self):
        with self._state_lock:
            return self._state

    def _set_app_state(self, state, tooltip=None):
        with self._state_lock:
            if state != self._state:
                self._state_since = time.time()
            self._state = state
        msg = tooltip or {
            AppState.LOADING: "Loading model...",
            AppState.IDLE: f"Ready [{self.engine.active_device_label}]",
            AppState.RECORDING: "Recording...",
            AppState.TRANSCRIBING: "Transcribing...",
            AppState.CLEANING: "Cleaning...",
            AppState.ERROR: "Startup failed",
        }[state]
        icon_state = {
            AppState.CLEANING: "transcribing",
        }.get(state, state.value)
        if hasattr(self, "icon"):
            self._set_tray(ICONS.get(icon_state, ICONS["idle"]), f"ZenVox - {msg}")
        self._queue_ui(lambda msg=msg: self.gui_status.set(msg))
        # Header status dot mirrors the tray color so the app window shows state too.
        dot_color = T.TRAY.get(icon_state, T.MUTED)
        self._queue_ui(lambda c=dot_color: self._status_dot.configure(text_color=c)
                       if hasattr(self, "_status_dot") else None)
        if hasattr(self, "_device_label"):
            self._queue_ui(lambda: self._device_label.configure(
                text=self.engine.active_device_label))
        self._queue_ui(self._update_rec_bar)

    # ── Watchdog ──────────────────────────────────────────────────────────
    def _watchdog_tick(self):
        """Periodic self-heal: un-stick the app if a state wedged. Runs on Tk."""
        try:
            self._check_stuck_state()
        except Exception:
            log.exception("Watchdog error")
        finally:
            try:
                self.root.after(3000, self._watchdog_tick)
            except RuntimeError:
                pass

    def _check_stuck_state(self):
        state = self._get_state()
        with self._state_lock:
            elapsed = time.time() - self._state_since
            deadline = self._pipeline_deadline
        # Dynamic deadline: long clips on the CPU fallback legitimately exceed
        # the fixed timeout; the pipeline sets a clip-scaled absolute deadline.
        stuck = elapsed > STUCK_PIPELINE_TIMEOUT
        if deadline is not None:
            stuck = time.time() > deadline
        if state in (AppState.TRANSCRIBING, AppState.CLEANING) and stuck:
            log.error(f"Watchdog: stuck in {state.value} for {elapsed:.0f}s — forcing reset")
            # Orphan the wedged task: bump the generation so its later writes
            # (state, paste, last_text) are suppressed, and replace the
            # executor so new stop tasks don't queue behind the hang.
            with self._state_lock:
                self._pipeline_gen += 1
            old = self._pipeline_executor
            self._pipeline_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="zenvox-pipeline")
            old.shutdown(wait=False, cancel_futures=True)
            self.engine.force_reset()
            self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}] - recovered from stall")
            self._overlay.hide()
            self._update_rec_bar()
        elif (state == AppState.RECORDING and self.engine.is_recording
                and self.engine.recording_duration > MAX_RECORD_SECONDS + 30):
            # Mic died mid-recording: callbacks stopped, so the in-callback
            # max-duration cap never fired. Salvage what was captured.
            log.warning("Watchdog: recording exceeded max duration + margin — forcing stop")
            self._stop_and_transcribe()
        elif (state == AppState.RECORDING and not self.engine.is_recording
                and not self._stopping and elapsed > 2.0):
            log.warning("Watchdog: state is RECORDING but engine is idle — resetting")
            self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}]")
            self._overlay.hide()
            self._update_rec_bar()

    def _pipeline_active(self):
        return self.engine.is_busy or self._get_state() in (AppState.TRANSCRIBING, AppState.CLEANING)

    def _guard_setting_change(self, label, revert=None):
        if not self._pipeline_active():
            return True
        log.warning(f"Ignoring {label} change while pipeline is active")
        # Tell the user WHY the widget snapped back — a silent revert reads
        # as a UI bug and gets retried in a loop.
        self._queue_ui(lambda: self.gui_status.set(
            f"Busy — change {label} after transcription finishes"))
        if revert is not None:
            self._queue_ui(revert)
        return False

    def _schedule_model_load(self, model_name):
        with self._state_lock:
            self._load_generation += 1
            load_id = self._load_generation
            self._pending_loads += 1
        self._set_app_state(AppState.LOADING, f"Loading {model_name}...")
        try:
            self._load_executor.submit(self._load_model_task, load_id, model_name)
        except RuntimeError:
            with self._state_lock:
                self._pending_loads = max(0, self._pending_loads - 1)
            self._set_app_state(AppState.ERROR, "Model loader is unavailable")

    def _load_model_task(self, load_id, model_name):
        try:
            self.engine.load_model(model_name=model_name)
        except Exception as exc:
            log.exception(f"Model load failed [{model_name}]")
            self._queue_ui(lambda load_id=load_id, model_name=model_name, exc=exc:
                           self._finish_model_load(load_id, model_name, exc))
            return
        self._queue_ui(lambda load_id=load_id, model_name=model_name:
                       self._finish_model_load(load_id, model_name, None))

    def _finish_model_load(self, load_id, model_name, exc):
        with self._state_lock:
            self._pending_loads = max(0, self._pending_loads - 1)
            latest_load = self._load_generation
            has_pending = self._pending_loads > 0

        if exc is not None:
            if load_id != latest_load or has_pending:
                self._set_app_state(AppState.LOADING, f"Loading {self.settings.model_name}...")
                return

            fallback_model = self.engine.loaded_model_name
            self.last_text = f"[Model load failed: {exc}]"
            self._gui_update_text(self.last_text)
            if fallback_model:
                if self.settings.model_name != fallback_model:
                    self.settings.model_name = fallback_model
                    self.settings.save()
                    self.gui_model.set(fallback_model)
                self._set_app_state(
                    AppState.IDLE,
                    f"Model load failed. Still using {fallback_model}.")
            else:
                self._set_app_state(AppState.ERROR, f"Model load failed: {exc}")
            return

        if load_id != latest_load or has_pending:
            self._set_app_state(AppState.LOADING, f"Loading {self.settings.model_name}...")
            return

        self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}]")

    # ── Settings Callbacks ────────────────────────────────────────────────
    def _on_model(self, v=None):
        new_model = self.gui_model.get()
        if not self._guard_setting_change(
                "model", revert=lambda: self.gui_model.set(self.settings.model_name)):
            return
        self.settings.model_name = new_model
        self.settings.save()
        self._schedule_model_load(new_model)
        self._rebuild_menu()  # tray radio checkmarks otherwise go stale

    def _on_lang(self, v=None):
        if not self._guard_setting_change(
                "language", revert=lambda: self.gui_lang.set(self.settings.lang_name)):
            return
        self.settings.lang_name = self.gui_lang.get()
        self.settings.save()
        self._rebuild_menu()

    def _on_mic(self, v=None):
        if not self._guard_setting_change(
                "microphone", revert=lambda: self.gui_mic.set(self.settings.mic_name)):
            return
        self.settings.mic_name = self.gui_mic.get()
        self.settings.save()
        self._rebuild_menu()

    def _on_provider(self, v=None):
        if not self._guard_setting_change(
                "provider", revert=lambda: self.gui_provider.set(self.settings.clean_provider)):
            return
        new_provider = self.gui_provider.get()
        self.settings.clean_provider = new_provider
        # Switch to the provider's default model
        default_model = PROVIDERS.get(new_provider, {}).get("default_model", "")
        self.settings.clean_model = default_model
        self.gui_clean.set(default_model)
        self._model_entry.configure(placeholder_text=default_model)
        # Ollama: show endpoint URL (unmasked) instead of an API key
        if new_provider in ("Ollama", "Local"):
            placeholder = ("http://localhost:11434/v1" if new_provider == "Ollama"
                           else "http://localhost:8001/v1")
            self._key_entry.configure(show="", placeholder_text=placeholder)
            self.gui_key.set(self.settings.ollama_endpoint if new_provider == "Ollama"
                             else self.settings.local_endpoint)
            self._key_entry.pack(side="left", padx=(0, 8), before=self._model_entry)
        else:
            needs_key = PROVIDERS.get(new_provider, {}).get("needs_key", True)
            self._key_entry.configure(show="*", placeholder_text="API key")
            self.gui_key.set(self.settings.get_api_key())
            if needs_key:
                self._key_entry.pack(side="left", padx=(0, 8), before=self._model_entry)
            else:
                self._key_entry.pack_forget()
        self.engine.invalidate_provider()
        self.settings.save()

    def _on_key(self):
        if not self._guard_setting_change(
                "credentials", revert=self._sync_provider_inputs):
            return
        provider = self.settings.clean_provider
        if provider == "Ollama":
            val = self.gui_key.get().strip()
            self.settings.ollama_endpoint = val or "http://localhost:11434/v1"
        elif provider == "Local":
            val = self.gui_key.get().strip()
            self.settings.local_endpoint = val or "http://localhost:8001/v1"
        else:
            self.settings.set_api_key(self.gui_key.get().strip())
        self.engine.invalidate_provider()
        self.settings.save()

    def _on_clean(self):
        if not self._guard_setting_change(
                "clean model", revert=lambda: self.gui_clean.set(self.settings.clean_model)):
            return
        self.settings.clean_model = self.gui_clean.get()
        self.engine.invalidate_provider()
        self.settings.save()

    def _on_silence(self, v=None):
        # Slider: live label update; persist without the busy-guard revert
        # (the slider can't be "wrong", just clamps).
        val = round(float(self.gui_silence.get()), 1)
        self._silence_val.configure(text=f"{val:.1f}s")
        self.settings.silence_timeout = val
        self.settings.save()

    def _on_output(self, v=None):
        if not self._guard_setting_change(
                "output mode", revert=lambda: self.gui_output.set(self.settings.output_mode)):
            return
        mode = self.gui_output.get()
        if mode == "Append to file" and not self.settings.output_file:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text", "*.txt"), ("All", "*.*")],
                title="Choose output file")
            if path:
                self.settings.output_file = path
            else:
                self.gui_output.set(self.settings.output_mode)
                return
        self.settings.output_mode = mode
        self.settings.save()
        self._sync_output_file_row()

    def _on_pick_output_file(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt"), ("All", "*.*")],
            title="Choose output file",
            initialfile=self.settings.output_file or "")
        if path:
            self.settings.output_file = path
            self.settings.save()
            self._sync_output_file_row()

    def _sync_output_file_row(self):
        """Show the append-to-file path + a Change button only in that mode —
        the path used to be un-viewable and un-changeable once first picked."""
        if not hasattr(self, "_output_file_label"):
            return
        if self.settings.output_mode == "Append to file":
            path = self.settings.output_file or "(no file chosen)"
            self._output_file_label.configure(text=f"→ {path}")
            self._output_file_label.master.pack_configure()
            self._output_file_btn.pack(side="right")
        else:
            self._output_file_label.configure(text="")
            self._output_file_btn.pack_forget()

    def _on_preset(self, v=None):
        if not self._guard_setting_change(
                "cleaning preset", revert=lambda: self.gui_preset.set(self.settings.cleaning_preset)):
            return
        self.settings.cleaning_preset = self.gui_preset.get()
        self.engine.invalidate_provider()
        self.settings.save()

    def _on_audio(self):
        self.settings.audio_feedback = self.gui_audio.get()
        self.settings.save()

    def _on_capture_mode(self, v=None):
        label = self.gui_capture.get()
        mode = next((k for k, lbl in self._capture_labels.items() if lbl == label), CAPTURE_TOGGLE)
        self.settings.capture_mode = mode
        self.settings.save()
        log.info(f"Capture mode -> {mode}")

    def _on_preview(self):
        self.settings.live_preview = bool(self.gui_preview.get())
        self.settings.save()

    def _on_quiet(self):
        self.settings.quiet_mode = bool(self.gui_quiet.get())
        self.settings.save()
        log.info(f"Quiet mode -> {self.settings.quiet_mode}")

    def _on_remote_token(self):
        val = self.gui_remote_token.get().strip()
        if val != self.settings.remote_token:
            self.settings.remote_token = val
            self.settings.save()

    def _on_test_remote(self):
        # Persist the current URL/token first, then probe off the Tk thread.
        self._on_remote_url()
        self._on_remote_token()
        self.gui_status.set("Testing remote server…")
        def _probe():
            ok = self.engine.remote_reachable(timeout=4.0)
            host = self.engine._remote_host()
            self._queue_ui(lambda: self.gui_status.set(
                f"Remote {host}: {'reachable ✓' if ok else 'unreachable ✕'}"))
        threading.Thread(target=_probe, daemon=True).start()

    # ── Per-app rules ─────────────────────────────────────────────────────
    def _refresh_rules(self):
        if not hasattr(self, "_rules_frame"):
            return
        for w in self._rules_frame.winfo_children():
            w.destroy()
        if not self.settings.app_rules:
            ctk.CTkLabel(self._rules_frame, text="No rules yet — the global cleaning style applies everywhere.",
                         font=T.F_SMALL, text_color=T.FAINT, anchor="w").pack(fill="x")
            return
        for i, rule in enumerate(self.settings.app_rules):
            row = ctk.CTkFrame(self._rules_frame, fg_color=T.PANEL2, corner_radius=T.R_CHIP)
            row.pack(fill="x", pady=2)
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=6)
            ctk.CTkLabel(inner, text=rule.get("pattern", ""), font=T.F_MONO_S,
                         text_color=T.TEXT, anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(inner, text="→ " + rule.get("preset", ""), font=T.F_SMALL,
                         text_color=T.BRAND).pack(side="left", padx=(8, 8))
            ctk.CTkButton(inner, text="✕", width=26, height=22, fg_color=T.PANEL,
                          hover_color=T.REC, text_color=T.MUTED, font=T.F_TINY,
                          corner_radius=T.R_CHIP,
                          command=lambda idx=i: self._on_rule_delete(idx)).pack(side="right")

    def _on_rule_add(self):
        pattern = self._rule_pattern.get().strip()
        if not pattern:
            return
        self.settings.app_rules.append(
            {"pattern": pattern, "preset": self.gui_rule_preset.get()})
        self.settings.save()
        self._rule_pattern.delete(0, "end")
        self._refresh_rules()

    def _on_rule_delete(self, idx):
        if 0 <= idx < len(self.settings.app_rules):
            self.settings.app_rules.pop(idx)
            self.settings.save()
            self._refresh_rules()

    def _on_rule_use_window(self, _seconds=3):
        """Prefill the pattern from the window the user wants a rule for — which
        is NOT this settings window. Count down so they can click the target
        window first, then read whatever is focused."""
        if _seconds > 0:
            self.gui_status.set(f"Click the target window — capturing in {_seconds}s…")
            self.root.after(1000, lambda: self._on_rule_use_window(_seconds - 1))
            return
        def _capture():
            info = _get_active_window_info()
            cls = (info.window_class or "").split()[0] if info.window_class else ""
            def _apply():
                if cls and "zenvox" not in cls.lower():
                    self._rule_pattern.delete(0, "end")
                    self._rule_pattern.insert(0, f"*{cls}*")
                    self.gui_status.set(f"Captured window: {cls}")
                elif cls:
                    self.gui_status.set("That's ZenVox — focus the app you want a rule for")
                else:
                    self.gui_status.set("Couldn't read the focused window (Wayland?)")
            self._queue_ui(_apply)
        threading.Thread(target=_capture, daemon=True).start()

    # ── Hotkey capture ────────────────────────────────────────────────────
    _CAPTURE_MODS = {"Control_L": "Ctrl", "Control_R": "Ctrl", "Alt_L": "Alt",
                     "Alt_R": "Alt", "ISO_Level3_Shift": "Alt", "Shift_L": "Shift",
                     "Shift_R": "Shift", "Super_L": "Win", "Super_R": "Win"}

    def _capture_hotkey(self, which):
        """Press-to-capture: grab the next key combo and store it as the hotkey.
        Armed only while the capture button holds keyboard focus; Escape or
        clicking away cancels and restores the previous value."""
        if getattr(self, "_hotkey_capturing", None):
            self._cancel_hotkey_capture()  # re-arm cleanly if already capturing
        btn = self._hotkey_rec_btn if which == "record" else self._hotkey_rep_btn
        var = self.gui_hotkey_record if which == "record" else self.gui_hotkey_repaste
        prev = var.get()
        var.set("Press keys… (Esc to cancel)")
        self._hotkey_capturing = which

        def finish():
            for seq, bid in self._hotkey_binds:
                try:
                    self.root.unbind(seq, bid)
                except Exception:
                    pass
            self._hotkey_binds = []
            self._hotkey_capturing = None

        def cancel(_=None):
            var.set(prev)
            finish()
            self.gui_status.set("Hotkey unchanged")
            return "break"

        def on_key(event):
            key = event.keysym
            if key == "Escape":
                return cancel()
            if key in self._CAPTURE_MODS:
                return "break"  # wait for the non-modifier key
            parts = []
            if event.state & 0x4: parts.append("Ctrl")
            if event.state & 0x8 or event.state & 0x80: parts.append("Alt")
            if event.state & 0x1: parts.append("Shift")
            combo = "+".join(parts + [key])
            finish()
            self._apply_hotkey(which, combo, prev, var)
            return "break"

        # Bind on the capture button (not the whole window), plus a FocusOut
        # cancel so clicking elsewhere doesn't leave capture armed forever.
        self._hotkey_binds = [
            ("<Key>", btn.bind("<Key>", on_key)),
            ("<FocusOut>", btn.bind("<FocusOut>", cancel)),
        ]
        btn.focus_set()

    def _cancel_hotkey_capture(self):
        for seq, bid in getattr(self, "_hotkey_binds", []):
            try:
                self.root.unbind(seq, bid)
            except Exception:
                pass
        self._hotkey_binds = []
        self._hotkey_capturing = None

    def _apply_hotkey(self, which, combo, prev, var):
        # Validate against the SAME normalization the live listener uses, so a
        # combo that parses but the listener can never match is rejected here.
        parsed = self._parse_hotkey_pynput(combo)
        if parsed is None:
            var.set(prev)
            self.gui_status.set(f"Can't use {combo} as a hotkey")
            return
        # A bare single letter/digit as a global hotkey would fire on every
        # keystroke — require a modifier or a function/special key.
        has_mod = any(m in combo for m in ("Ctrl", "Alt", "Shift", "Win"))
        base = combo.split("+")[-1].lower()
        is_special = base.startswith("f") and base[1:].isdigit() or base in (
            "space", "insert", "delete", "home", "end", "pageup", "pagedown", "pause")
        if not has_mod and not is_special:
            var.set(prev)
            self.gui_status.set(f"{combo} needs a modifier (e.g. Ctrl+{combo})")
            return
        var.set(combo)
        if which == "record":
            self.settings.hotkey_record = combo
        else:
            self.settings.hotkey_repaste = combo
        self.settings.save()
        # Respawn the listener so the new binding takes effect immediately.
        self._restart_hotkey_listener()
        if hasattr(self, "_hint_label"):
            self._hint_label.configure(
                text=f"Press  {self.settings.hotkey_record.lower()}  to dictate into any app")
        self.gui_status.set(f"Hotkey set: {combo}")

    def _restart_hotkey_listener(self):
        # Bump the generation FIRST so any run-loop currently in its backoff
        # sleep exits instead of re-entering _hotkey_listener — otherwise a
        # restart during that window would leave two live listeners.
        self._hotkey_gen += 1
        listener = getattr(self, "_active_listener", None)
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        threading.Thread(target=self._run_hotkey_listener,
                         args=(self._hotkey_gen,), daemon=True).start()

    def _on_backend(self, v=None):
        if not self._guard_setting_change(
                "backend", revert=lambda: self.gui_backend.set(
                    self._backend_labels.get(self.settings.transcription_backend, ""))):
            return
        label = self.gui_backend.get()
        backend = next((k for k, lbl in self._backend_labels.items() if lbl == label), BACKEND_LOCAL)
        if backend == self.settings.transcription_backend:
            return
        self.settings.transcription_backend = backend
        self.settings.save()
        log.info(f"Transcription backend -> {backend}")
        # Reload: remote unloads the local Whisper model, local loads it back.
        self._schedule_model_load(self.settings.model_name)

    def _on_remote_url(self):
        url = self.gui_remote_url.get().strip()
        if url and url != self.settings.remote_url:
            self.settings.remote_url = url
            self.settings.save()
            log.info(f"Remote ASR URL -> {url}")

    def _sync_provider_inputs(self):
        provider = self.settings.clean_provider
        self.gui_provider.set(provider)
        default_model = PROVIDERS.get(provider, {}).get("default_model", "")
        self._model_entry.configure(placeholder_text=default_model)
        self.gui_clean.set(self.settings.clean_model)
        if provider in ("Ollama", "Local"):
            placeholder = ("http://localhost:11434/v1" if provider == "Ollama"
                           else "http://localhost:8001/v1")
            self._key_entry.configure(show="", placeholder_text=placeholder)
            self.gui_key.set(self.settings.ollama_endpoint if provider == "Ollama"
                             else self.settings.local_endpoint)
            self._key_entry.pack(side="left", padx=(0, 8), before=self._model_entry)
        else:
            needs_key = PROVIDERS.get(provider, {}).get("needs_key", True)
            self._key_entry.configure(show="*", placeholder_text="API key")
            self.gui_key.set(self.settings.get_api_key())
            if needs_key:
                self._key_entry.pack(side="left", padx=(0, 8), before=self._model_entry)
            else:
                self._key_entry.pack_forget()

    # ── Recording bar ─────────────────────────────────────────────────────
    def _update_rec_bar(self):
        # Cancel any pending tick first \u2014 _set_app_state also queues this
        # method, and duplicate 100ms chains used to pile up during recording.
        if self._timer_job is not None:
            try:
                self.root.after_cancel(self._timer_job)
            except Exception:
                pass
            self._timer_job = None
        state = self._get_state()
        if state == AppState.RECORDING and self.engine.is_recording:
            self._level_bar.set(self.engine.audio_level)
            self._level_bar.configure(progress_color=T.REC)
            self._overlay.set_level(self.engine.audio_level)
            d = self.engine.recording_duration
            timer_str = f"{int(d // 60):02d}:{d % 60:04.1f}"
            self._timer_label.configure(text=f"Recording  {timer_str}", text_color=T.REC)
            # Button shows stop icon during recording
            self._rec_btn.configure(text="\u25a0", fg_color=T.REC_HOVER,
                                    hover_color=T.REC, state="normal")
            self._overlay.update_timer(f"{int(d // 60)}:{int(d % 60):02d}")
            self._timer_job = self.root.after(100, self._update_rec_bar)
        elif state == AppState.RECORDING or self._stopping:
            # Stop/transcribe handoff: engine already idle but the stop task is
            # still joining workers. The old code fell through to "Ready" and
            # re-enabled the record button \u2014 clicking it clobbered the
            # in-flight dictation.
            self._level_bar.set(0)
            self._timer_label.configure(text="Finishing\u2026", text_color=T.MUTED)
            self._rec_btn.configure(text="\u25cf", fg_color=T.PANEL2,
                                    hover_color=T.PANEL2, state="disabled")
            self._timer_job = self.root.after(100, self._update_rec_bar)
        elif state in (AppState.TRANSCRIBING, AppState.CLEANING):
            self._level_bar.set(0)
            label = "Cleaning\u2026" if state == AppState.CLEANING else "Transcribing\u2026"
            self._timer_label.configure(text=label, text_color=T.AMBER)
            self._rec_btn.configure(text="\u25cf", fg_color=T.PANEL2,
                                    hover_color=T.PANEL2, state="disabled")
        elif state == AppState.LOADING:
            self._level_bar.set(0)
            self._timer_label.configure(text="Loading model\u2026", text_color=T.MUTED)
            self._rec_btn.configure(text="\u25cf", fg_color=T.PANEL2,
                                    hover_color=T.PANEL2, state="disabled")
        elif state == AppState.ERROR:
            self._level_bar.set(0)
            self._timer_label.configure(text="Model load failed", text_color=T.REC)
            self._rec_btn.configure(text="\u25cf", fg_color=T.PANEL2,
                                    hover_color=T.PANEL2, state="disabled")
        else:
            self._level_bar.set(0)
            self._timer_label.configure(text="Ready", text_color=T.TEXT)
            self._rec_btn.configure(text="\u25cf", fg_color=T.REC,
                                    hover_color=T.REC_HOVER, state="normal")
            self._refresh_stats()

    # ── Tray Menu ─────────────────────────────────────────────────────────
    def _build_menu(self):
        def model_action(n):
            def a(icon, item):
                if self._pipeline_active():
                    log.warning("Ignoring tray model change while pipeline is active")
                    return
                self.settings.model_name = n
                self.settings.save()
                self.root.after(0, lambda: self.gui_model.set(n))
                self._schedule_model_load(n)
            return a

        def lang_action(n):
            def a(icon, item):
                if self._pipeline_active():
                    log.warning("Ignoring tray language change while pipeline is active")
                    return
                self.settings.lang_name = n
                self.settings.save()
                self.root.after(0, lambda: self.gui_lang.set(n))
            return a

        def mic_action(n):
            def a(icon, item):
                if self._pipeline_active():
                    log.warning("Ignoring tray microphone change while pipeline is active")
                    return
                self.settings.mic_name = n
                self.settings.save()
                self.root.after(0, lambda: self.gui_mic.set(n))
            return a

        models = [pystray.MenuItem(m, model_action(m),
                  checked=lambda item, m=m: self.settings.model_name == m,
                  radio=True) for m in MODELS]
        langs = [pystray.MenuItem(l, lang_action(l),
                 checked=lambda item, l=l: self.settings.lang_name == l,
                 radio=True) for l in LANGS]
        mics = [pystray.MenuItem(n, mic_action(n),
                checked=lambda item, n=n: self.settings.mic_name == n,
                radio=True) for _, n in self.input_devs]

        return pystray.Menu(
            pystray.MenuItem(
                lambda item: (f"Last: {self.last_text[:50]}..."
                              if len(self.last_text) > 50
                              else f"Last: {self.last_text}"
                              if self.last_text else "Last: -"),
                self._copy_last),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Window", lambda i, it: self._show_window(), default=True),
            pystray.MenuItem("Re-paste last",
                             lambda i, it: threading.Thread(
                                 target=self._repaste, daemon=True).start()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Model", pystray.Menu(*models)),
            pystray.MenuItem("Language", pystray.Menu(*langs)),
            pystray.MenuItem("Mic", pystray.Menu(*mics)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    def _copy_last(self, icon, item):
        self._copy_to_clipboard(self.last_text)

    def _hide_window(self):
        self.root.withdraw()

    def _quit_from_ui(self):
        self._quit(None, None)

    def _show_window(self):
        def _raise():
            self.root.deiconify()
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.focus_force()
            # Remove topmost after raising so it doesn't stay pinned
            self.root.after(100, lambda: self.root.attributes('-topmost', False))
        self.root.after(0, _raise)

    def _quit(self, icon, item):
        self._shutting_down = True
        self.history.close()
        self._load_executor.shutdown(wait=False, cancel_futures=True)
        self._pipeline_executor.shutdown(wait=False, cancel_futures=True)
        _cleanup_clipboard_owners(list(_PERSISTENT_SELECTION_OWNERS.values()))
        _PERSISTENT_SELECTION_OWNERS.clear()
        self.icon.stop()
        self.root.after(0, self.root.quit)
        # A non-daemon pipeline/loader thread mid-flight (e.g. a 60s remote
        # POST) keeps the process alive after the UI is gone. History and
        # settings are already persisted above; force-exit after a grace period.
        t = threading.Timer(2.0, lambda: os._exit(0))
        t.daemon = True
        t.start()

    def _run_hotkey_listener(self, gen=0):
        # One pynput/X error (suspend/resume, Xorg hiccup, XRecord failure)
        # used to kill this thread permanently — tray up, hotkeys silently
        # dead. Restart with backoff; only give up after repeated failures.
        # gen: a hotkey change bumps _hotkey_gen and spawns a fresh loop; a
        # superseded loop (stale gen) must exit rather than re-enter and leave
        # two live listeners running.
        failures = 0
        while not self._shutting_down and gen == self._hotkey_gen:
            started = time.time()
            try:
                self._hotkey_listener(gen)
                return  # listener exited cleanly (shutdown or superseded)
            except Exception:
                log.exception("Hotkey listener crashed")
            if gen != self._hotkey_gen:
                return  # superseded during the run — newer loop owns it now
            if time.time() - started > 60:
                failures = 0  # ran fine for a while — treat as a fresh crash
            failures += 1
            if failures >= 5:
                log.error("Hotkey listener failed 5 times — giving up")
                self._queue_ui(lambda: self.gui_status.set(
                    "Hotkeys unavailable — restart ZenVox"))
                return
            # Interruptible backoff: wake early if superseded so we don't hold a
            # stale loop alive for up to 8s.
            for _ in range(int(2 * failures * 10)):
                if self._shutting_down or gen != self._hotkey_gen:
                    return
                time.sleep(0.1)

    # ── Hotkey ────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_hotkey_pynput(hotkey_str):
        """Parse 'Ctrl+Alt+F12' into a set of pynput keys."""
        from pynput.keyboard import Key, KeyCode
        MODIFIER_MAP = {
            "ctrl": Key.ctrl_l, "alt": Key.alt_l,
            "shift": Key.shift_l, "win": Key.cmd,
        }
        KEY_MAP = {
            "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4,
            "f5": Key.f5, "f6": Key.f6, "f7": Key.f7, "f8": Key.f8,
            "f9": Key.f9, "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
            "space": Key.space, "enter": Key.enter, "tab": Key.tab,
            "insert": Key.insert, "delete": Key.delete,
            "home": Key.home, "end": Key.end,
            "pageup": Key.page_up, "pagedown": Key.page_down,
            "pause": Key.pause,
            "media_play_pause": Key.media_play_pause,
            "media_next": Key.media_next,
            "media_previous": Key.media_previous,
            "media_volume_up": Key.media_volume_up,
            "media_volume_down": Key.media_volume_down,
            "media_volume_mute": Key.media_volume_mute,
        }
        parts = [p.strip().lower() for p in hotkey_str.split("+") if p.strip()]
        keys = set()
        for p in parts:
            if p in MODIFIER_MAP:
                keys.add(MODIFIER_MAP[p])
            elif p in KEY_MAP:
                keys.add(KEY_MAP[p])
            elif len(p) == 1 and p.isalnum():
                keys.add(KeyCode.from_char(p))
            else:
                # An unknown token silently dropped would leave a PARTIAL combo
                # ("Ctrl+F13" -> just Ctrl) that fires on a bare modifier press.
                log.error(f"Hotkey {hotkey_str!r}: unknown key {p!r} — hotkey disabled")
                return None
        return frozenset(keys) if keys else None

    def _hotkey_listener(self, gen=0):
        from pynput.keyboard import Listener

        if gen != self._hotkey_gen:
            return  # superseded before we even started

        if (sys.platform.startswith("linux")
                and _linux_session_type() == "wayland"):
            # pynput cannot grab global keys on native Wayland; be honest
            # instead of listening to nothing.
            log.warning("Wayland session: global hotkeys may not work — "
                        "use the window/tray to record")
            self._queue_ui(lambda: self.gui_status.set(
                "Wayland: global hotkeys may not work — use the Record button"))

        rec_combo = self._parse_hotkey_pynput(self.settings.hotkey_record)
        rep_combo = self._parse_hotkey_pynput(self.settings.hotkey_repaste)
        if rec_combo is None:
            # Bad user-configured combo: fall back to the default so dictation
            # still works, and say so.
            default = Settings.__dataclass_fields__["hotkey_record"].default
            rec_combo = self._parse_hotkey_pynput(default)
            self._queue_ui(lambda: self.gui_status.set(
                f"Bad record hotkey in settings — using {default}"))
        current_keys = set()
        ptt = {"timer": None}
        PTT_GRACE = 0.18  # window to absorb X11 auto-repeat (press/release bursts)

        def cancel_timer():
            t = ptt["timer"]
            if t is not None:
                t.cancel()
                ptt["timer"] = None

        def ptt_stop():
            ptt["timer"] = None
            if self.engine.is_recording:
                self.root.after(0, self._stop_and_transcribe)

        listener_ref = [None]

        def canon(key):
            # Map a modifier-affected key back to its base (e.g. Ctrl+Alt+K may
            # arrive as a distinct KeyCode) so a modifier+letter combo actually
            # matches KeyCode.from_char('k'). pynput's documented pattern.
            lst = listener_ref[0]
            if lst is not None:
                try:
                    return lst.canonical(key)
                except Exception:
                    pass
            return key

        def on_press(key):
            # Runs on pynput's listener thread. An uncaught exception here KILLS
            # the listener — the tray stays up but the record hotkey goes dead.
            # So: never do real work here, never raise. Work runs on its own
            # short-lived thread (off both the listener and the Tk main loop).
            try:
                current_keys.add(canon(key))
                if rec_combo and rec_combo.issubset(current_keys):
                    if self.settings.capture_mode == CAPTURE_PTT:
                        # A press during the grace window is auto-repeat, not a
                        # real re-press — cancel any pending stop and keep going.
                        cancel_timer()
                        threading.Thread(target=self._safe_start_ptt, daemon=True).start()
                    else:
                        current_keys.clear()
                        threading.Thread(target=self._safe_toggle, daemon=True).start()
                elif rep_combo and rep_combo.issubset(current_keys):
                    current_keys.clear()
                    threading.Thread(target=self._repaste, daemon=True).start()
            except Exception:
                log.exception("Hotkey on_press error (ignored to keep listener alive)")

        def on_release(key):
            try:
                key = canon(key)
                was_rec_key = key in rec_combo
                current_keys.discard(key)
                if (self.settings.capture_mode == CAPTURE_PTT and was_rec_key
                        and self.engine.is_recording):
                    # Defer the stop briefly: X11 auto-repeat fires release+press
                    # rapidly while held, so only stop if no re-press follows.
                    cancel_timer()
                    t = threading.Timer(PTT_GRACE, ptt_stop)
                    t.daemon = True
                    ptt["timer"] = t
                    t.start()
            except Exception:
                log.exception("Hotkey on_release error (ignored)")

        log.info(f"Hotkeys: {self.settings.hotkey_record}=record ({self.settings.capture_mode}), "
                 f"{self.settings.hotkey_repaste}=re-paste")
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener_ref[0] = listener        # for canonical() in the handlers
            self._active_listener = listener  # so a hotkey change can stop + respawn it
            listener.join()

    def _safe_toggle(self):
        """Toggle wrapper for the hotkey thread — never lets a failure escape."""
        try:
            self._toggle()
        except Exception:
            log.exception("Toggle failed — recovering to IDLE")
            self.engine.force_reset()
            self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}] - recovered")
            self.root.after(0, self._overlay.hide)
            self.root.after(0, self._update_rec_bar)

    def _safe_start_ptt(self):
        """Push-to-talk start (record key pressed/held). Begins recording if idle."""
        try:
            state = self._get_state()
            if state in (AppState.LOADING, AppState.TRANSCRIBING, AppState.CLEANING, AppState.ERROR):
                return
            if self._stopping:
                return  # previous utterance still releasing the stream
            if self.engine.is_recording or not self.engine.is_ready:
                return
            self._start_recording()
        except Exception:
            log.exception("PTT start failed — recovering")
            self.engine.force_reset()
            self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}] - recovered")
            self.root.after(0, self._overlay.hide)
            self.root.after(0, self._update_rec_bar)

    def _toggle(self):
        state = self._get_state()
        if state in (AppState.LOADING, AppState.TRANSCRIBING, AppState.CLEANING, AppState.ERROR):
            return
        if self._stopping:
            return  # stop/transcribe handoff in flight — not a valid start window
        # Atomic decision: debounce + readiness + start-vs-stop under one engine
        # lock, so the GUI button and the hotkey can't both decide to start.
        action = self.engine.request_toggle()
        if action == "start":
            self._start_recording()
        elif action == "stop":
            self._stop_and_transcribe()

    def _repaste(self):
        with self._clipboard_lock:
            if not self.last_text:
                return
            target = _get_active_window_info()
            owners = _copy_text(
                self.last_text,
                include_primary=sys.platform.startswith("linux"),
                for_immediate_paste=True,
            )
            # Keep the selection owners ALIVE after pasting — killing them
            # right away emptied the clipboard on Wayland (wl-copy owns the
            # selection for as long as it runs) and raced slow X11 apps that
            # fetch the selection on demand. They're replaced on the next copy.
            for owner in owners:
                prior = _PERSISTENT_SELECTION_OWNERS.get(owner.selection)
                _close_clipboard_owner(prior)
                _PERSISTENT_SELECTION_OWNERS[owner.selection] = owner
            time.sleep(0.05 if sys.platform == "win32" else 0.2)
            _simulate_paste(window=target)
            log.info("Re-pasted")

    # ── Recording Flow ────────────────────────────────────────────────────
    def _start_recording(self):
        # Capture the active window id BEFORE the overlay shows (one subprocess,
        # ~10ms); the slower class/name lookups fill in on a background thread —
        # they're only needed at clean/paste time, and doing all three serially
        # here delayed the mic open enough to clip the first syllables.
        self._paste_target_window = _get_active_window_id_fast()
        threading.Thread(target=_fill_window_details,
                         args=(self._paste_target_window,), daemon=True).start()
        dev_id = self.engine.get_device_id(self.input_devs)
        # In push-to-talk, key-release controls the stop, so disable VAD silence
        # auto-stop. on_vad_stop stays wired for the max-duration safety cap.
        ptt = self.settings.capture_mode == CAPTURE_PTT
        try:
            self.engine.start_recording(
                device_id=dev_id, vad_autostop=not ptt,
                on_vad_stop=lambda: self.root.after(0, self._stop_and_transcribe),
                on_partial=lambda t: self.root.after(0, lambda t=t: self._on_partial_preview(t)))
            self._set_app_state(AppState.RECORDING, "Recording (hold)..." if ptt else "Recording...")
            self.root.after(0, lambda: self._overlay.show("recording"))
            self.root.after(0, self._update_rec_bar)
        except Exception as e:
            log.exception(f"Start failed: {e}")
            self._set_app_state(AppState.IDLE, "Mic error - check device")
            self.root.after(0, self._update_rec_bar)

    def _stop_and_transcribe(self):
        # Called from the VAD auto-stop (Tk thread) and the hotkey thread. The
        # stop itself joins the VAD worker (up to 2s); doing that on either of
        # those threads freezes the UI or risks the listener. Run the whole
        # stop+transcribe on the single-worker pipeline executor instead.
        gen = self._pipeline_gen
        try:
            self._pipeline_executor.submit(self._stop_and_transcribe_task, gen)
        except RuntimeError:
            log.exception("Pipeline executor is unavailable")
            self.engine.force_reset()  # release stream + VAD worker even when gone
            self._set_app_state(AppState.IDLE, "Pipeline unavailable - try again")
            self.root.after(0, self._overlay.hide)

    def _pipeline_current(self, gen):
        """False once the watchdog has orphaned this task — a stale task must
        not touch state, paste, or last_text anymore."""
        return gen == self._pipeline_gen

    def _stop_and_transcribe_task(self, gen):
        # _stopping marks the window where engine._recording is already False but
        # the app state is still RECORDING (stop_recording joins the VAD worker
        # and closes the stream, up to ~2s). Without it, the watchdog would see
        # "RECORDING but engine idle" and wrongly flash IDLE / hide the overlay.
        self._stopping = True
        try:
            try:
                audio, duration = self.engine.stop_recording()
            except Exception:
                log.exception("stop_recording failed — recovering to IDLE")
                self.engine.force_reset()
                self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}] - recovered")
                self.root.after(0, self._overlay.hide)
                self.root.after(0, self._update_rec_bar)
                return
            if not self._pipeline_current(gen):
                return
            self.root.after(0, self._update_rec_bar)
            if audio is None:
                self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}]")
                self.root.after(0, self._overlay.hide)
                return
            self._last_clean_reason = ""
            # Long clips on the CPU fallback legitimately take minutes — give
            # the watchdog a clip-scaled deadline instead of the fixed 90s.
            with self._state_lock:
                self._pipeline_deadline = time.time() + max(
                    STUCK_PIPELINE_TIMEOUT, duration * 3 + 60)
            self._set_app_state(AppState.TRANSCRIBING, "Transcribing...")
            self.root.after(0, lambda: self._overlay.show("transcribing"))
            self.root.after(0, self._update_rec_bar)
            self._transcribe(audio, duration, gen)
        finally:
            # Only clear shared state if WE are still the current pipeline task.
            # A watchdog reset bumps the generation and starts a fresh task; an
            # orphaned task unwedging later must not clear the new task's
            # _stopping / _pipeline_deadline out from under it.
            if self._pipeline_current(gen):
                self._stopping = False
                with self._state_lock:
                    self._pipeline_deadline = None

    def _snapshot_clipboard_async(self, gen):
        """Read the current clipboard state in the background (during cleaning)
        so the paste path doesn't pay 2 serial xclip reads — or a hung clipboard
        owner's 5s timeout — on the critical path. Generation-tagged so a slow
        reader from a previous dictation can't overwrite this one's snapshot."""
        def _read():
            state = _read_clipboard_state(timeout=1.5)
            with self._clipboard_lock:
                if self._pipeline_current(gen):
                    self._prepaste_state = state
        self._prepaste_state = None
        threading.Thread(target=_read, daemon=True).start()

    def _transcribe(self, audio, duration, gen):
        final_msg = None  # a terminal message the finally must NOT overwrite
        try:
            t0 = time.monotonic()
            try:
                raw = self.engine.transcribe(
                    audio,
                    on_segment=lambda t: self.root.after(
                        0, lambda t=t: self._gui_update_text(t)))
            except TranscriptionError as e:
                # Dictation LOST — say so. The old code returned "" here and
                # flashed back to Ready as if nothing happened.
                final_msg = f"Transcription failed: {e}"
                return
            t_transcribed = time.monotonic()
            if not raw:
                return
            if not self._pipeline_current(gen):
                return
            # Per-app preset: pick the cleaning style that matches the window
            # the user dictated into (Slack→casual, Gmail→Email, editor→Technical).
            target = self._paste_target_window
            preset = self.settings.cleaning_preset
            if isinstance(target, ActiveWindowInfo) and self.settings.app_rules:
                preset = self.settings.preset_for_window(
                    target.window_class, target.window_name)
                if preset != self.settings.cleaning_preset:
                    log.info(f"Per-app rule: {target.window_class!r} -> {preset} preset")
            self._set_app_state(
                AppState.CLEANING,
                f"Cleaning [{preset}]...")
            self.root.after(0, lambda: self._overlay.set_label("Cleaning"))
            # Overlap the clipboard snapshot with the LLM round-trip.
            self._snapshot_clipboard_async(gen)
            clean_result = self.engine.clean_text(raw, preset=preset)
            t_cleaned = time.monotonic()
            if clean_result.used_fallback and clean_result.reason:
                self._last_clean_reason = clean_result.reason
                log.warning(f"Using raw transcription fallback: {clean_result.reason}")
            else:
                self._last_clean_reason = ""
            text = clean_result.text
            if not self._pipeline_current(gen):
                log.warning("Stale pipeline task finished — discarding result (watchdog reset)")
                return
            self.last_text = text
            self.root.after(0, lambda t=text: self._gui_update_text(t))

            # Output FIRST — history/menu bookkeeping must never delay or
            # destroy a successful dictation (a locked history.db used to eat
            # the paste and clobber last_text).
            # Hide the overlay BEFORE paste — its -topmost flag intercepts the
            # paste keystroke on X11, so nothing reaches the target window.
            self.root.after(0, self._overlay.hide)
            self._output_text(text, gen)
            t_pasted = time.monotonic()
            self._last_latency = (
                int((t_transcribed - t0) * 1000),
                int((t_cleaned - t_transcribed) * 1000),
                int((t_pasted - t_cleaned) * 1000))
            log.info("Latency: transcribe=%dms clean=%dms paste=%dms total=%dms (clip %.1fs)"
                     % (*self._last_latency, int((t_pasted - t0) * 1000), duration))

            try:
                self.history.add(
                    raw_text=raw, cleaned_text=text,
                    language=self.settings.lang_name,
                    duration_sec=round(duration, 1),
                    model=self.settings.model_name,
                    cleaning_preset=preset)  # the effective (per-app) preset, not the global one
            except Exception:
                log.exception("history.add failed — dictation already delivered")
            try:
                self._rebuild_menu()
            except Exception:
                log.exception("tray menu rebuild failed")
            if self._current_view == "history":
                self.root.after(0, self._refresh_history)
        except Exception as e:
            log.exception(f"Pipeline: {e}")
            if self._pipeline_current(gen):
                self.last_text = f"[Error: {e}]"
                self.root.after(0, lambda: self._gui_update_text(self.last_text))
                final_msg = f"Pipeline error: {e}"
        finally:
            if self._pipeline_current(gen):
                if final_msg is not None:
                    ready_msg = final_msg  # keep the error visible, don't flash "Ready"
                elif self._last_clean_reason:
                    ready_msg = f"Ready [{self.engine.active_device_label}] - raw text kept ({self._last_clean_reason})"
                else:
                    ready_msg = f"Ready [{self.engine.active_device_label}]"
                self._set_app_state(AppState.IDLE, ready_msg)
                self.root.after(0, self._overlay.hide)
                self.root.after(0, self._update_rec_bar)

    def _set_clipboard_tk(self, text):
        """Set the X11 clipboard via Tk's native API (thread-safe).

        Tk is already running its mainloop and can own the X11 clipboard
        selection natively — no xclip subprocess, no fork, no timeout.
        Scheduled on the Tk thread via root.after; blocks the caller until done.
        """
        done = threading.Event()
        ok = [False]  # only _do's success sets True — a wait timeout is NOT success
        cancelled = [False]  # set on timeout so a late _do doesn't clobber the clipboard
        def _do():
            if cancelled[0]:
                return  # lost the race — the xclip fallback + restore already ran
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update()
                ok[0] = True
            except Exception as e:
                log.error(f"Tk clipboard set failed: {e}")
            finally:
                done.set()
        try:
            self.root.after(0, _do)
            if not done.wait(timeout=2.0):
                # Tk main loop stalled — pasting now would paste STALE contents.
                # Mark the queued _do cancelled so it doesn't fire later and
                # overwrite the clipboard the xclip fallback + restore just set.
                cancelled[0] = True
                log.warning("Tk clipboard set timed out — falling back to xclip")
        except RuntimeError:
            pass  # root was destroyed — fall through to xclip
        if not ok[0]:
            owners = _copy_text(text, for_immediate_paste=True)
            for owner in owners:
                prior = _PERSISTENT_SELECTION_OWNERS.get(owner.selection)
                _close_clipboard_owner(prior)
                _PERSISTENT_SELECTION_OWNERS[owner.selection] = owner
            ok[0] = True  # xclip owner path either worked or logged its own failure
        return ok[0]

    def _output_text(self, text, gen=None):
        mode = self.settings.output_mode
        if gen is not None and not self._pipeline_current(gen):
            return
        if mode == "Auto-paste":
            target = self._paste_target_window
            paste_text = text.rstrip("\n")
            # A terminal executes every newline in a paste as a command — never
            # auto-type multi-line text into one. Collapse to a single line;
            # the untouched text still lands in history and last_text.
            if _is_terminal_window(target) and "\n" in paste_text:
                log.warning("Paste target is a terminal — collapsing newlines to spaces")
                paste_text = " ".join(line.strip() for line in paste_text.splitlines() if line.strip())
            with self._clipboard_lock:
                # Snapshot was taken in the background during cleaning; fall
                # back to a tight read if it didn't land.
                prev = self._prepaste_state or _read_clipboard_state(timeout=1.5)
                self._prepaste_state = None
                # Set clipboard via Tk native API — bypasses xclip entirely,
                # which was timing out and failing on this Ubuntu/X11 setup.
                clipboard_ok = self._set_clipboard_tk(paste_text)
                # Also set primary selection (middle-click paste) via xclip —
                # Tk has no primary-selection API, so this stays subprocess-based.
                if sys.platform.startswith("linux"):
                    _copy_selection_linux(paste_text, "primary")
                try:
                    # Tk clipboard is immediate (no fork to wait for), but a
                    # brief settle lets the overlay withdraw + X11 process the
                    # selection transfer before the keystroke lands.
                    paste_delay = 0.05 if sys.platform == "win32" else 0.12
                    time.sleep(paste_delay)
                    if clipboard_ok:
                        _simulate_paste(window=target)
                    else:
                        log.error("Clipboard could not be set — skipping paste keystroke "
                                  "(would have pasted stale contents)")
                except Exception as e:
                    log.exception(f"Auto-paste failed: {e}")

            # Restore the previous clipboard in the background — the user's
            # text is already in the target app; holding the hotkey hostage
            # for the 0.8s settle + 2 xclip reads was pure dead time.
            def _restore():
                time.sleep(0.15 if sys.platform == "win32" else 0.7)
                with self._clipboard_lock:
                    try:
                        _restore_clipboard_state(prev, expected_text=paste_text,
                                                 last_pasted_text=self._last_pasted)
                        self._last_pasted = paste_text
                    except Exception:
                        log.exception("Clipboard restore failed")
            threading.Thread(target=_restore, daemon=True).start()
        elif mode == "Clipboard only":
            self._set_clipboard_tk(text)
            log.info("Clipboard mode — set via Tk")
        elif mode == "Append to file":
            path = self.settings.output_file
            if path:
                try:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- {datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
                        f.write(text + "\n")
                    log.info(f"Appended to {path}")
                except Exception as e:
                    log.error(f"File write: {e}")
                    self._set_clipboard_tk(text)
            else:
                self._set_clipboard_tk(text)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    try:
        ZenVoxApp()
    except Exception:
        log.exception("ZenVox failed to start")
        raise


if __name__ == "__main__":
    main()
