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
from providers import PROVIDERS, PROVIDER_NAMES, create_provider
from history import History
from dictionary import Dictionary, DictionaryEntry

log = setup_logging()


def _patch_pystray_appindicator_icons():
    """Teach pystray's AppIndicator backend to use a real icon name + theme path."""
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


_patch_pystray_appindicator_icons()


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


def _read_selection_linux(selection):
    session_type = _linux_session_type()
    if session_type == "wayland" and shutil.which("wl-paste"):
        args = ["wl-paste", "--no-newline"]
        if selection == "primary":
            args.append("--primary")
        result = _run_text_command(args)
        return result.stdout if result.returncode == 0 else ""
    if shutil.which("xclip"):
        result = _run_text_command(["xclip", "-selection", selection, "-o"])
        return result.stdout if result.returncode == 0 else ""
    if shutil.which("xsel"):
        flag = "--clipboard" if selection == "clipboard" else "--primary"
        result = _run_text_command(["xsel", flag, "--output"])
        return result.stdout if result.returncode == 0 else ""
    return ""


def _read_clipboard_state():
    if sys.platform == "win32":
        try:
            return ClipboardState(clipboard=pyperclip.paste())
        except Exception:
            return ClipboardState()

    clipboard = _read_selection_linux("clipboard")
    primary = None
    if sys.platform.startswith("linux"):
        primary = _read_selection_linux("primary")
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


def _restore_clipboard_state(previous, expected_text, last_pasted_text=""):
    current = _read_clipboard_state()
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
    if not isinstance(window, ActiveWindowInfo):
        return False
    haystack = f"{window.window_class} {window.window_name}".lower()
    return any(marker in haystack for marker in _TERMINAL_CLASS_MARKERS)


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


def _get_active_window_info():
    """Get the currently focused window metadata on Linux/X11."""
    if sys.platform == "win32":
        return ActiveWindowInfo()
    if _linux_session_type() == "wayland":
        return ActiveWindowInfo()
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            window_id = result.stdout.strip()
            return ActiveWindowInfo(
                window_id=window_id,
                window_class=_get_window_class(window_id),
                window_name=_get_window_name(window_id),
            )
    except Exception:
        pass
    return ActiveWindowInfo()


def _simulate_paste(window=None):
    """Simulate paste on the current platform."""
    if sys.platform == "win32":
        pyautogui.hotkey("ctrl", "v")
    else:
        session_type = _linux_session_type()
        if session_type == "wayland":
            if shutil.which("wtype"):
                subprocess.run(["wtype", "-M", "shift", "Insert", "-m", "shift"], check=False)
                return
            pyautogui.hotkey("ctrl", "v")
            return

        window_id = window.window_id if isinstance(window, ActiveWindowInfo) else window
        if window_id:
            try:
                subprocess.run(["xdotool", "windowactivate", "--sync", window_id],
                               timeout=2, check=False)
                subprocess.run(["xdotool", "windowfocus", "--sync", window_id],
                               timeout=2, check=False)
                time.sleep(0.1)
            except Exception:
                pass

        # Terminals on Ubuntu usually need Ctrl+Shift+V; most other apps expect Ctrl+V.
        key_combo = "ctrl+shift+v" if _is_terminal_window(window) else "ctrl+v"
        if shutil.which("xdotool"):
            subprocess.run(["xdotool", "key", "--clearmodifiers", key_combo], check=False)
        else:
            pyautogui.hotkey(*key_combo.split("+"))


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


def _post_multipart_audio(url, wav_bytes, fields, timeout=60.0):
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
    req = _req.Request(url, data=body, method="POST",
                       headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
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
            with self._lock:
                self._whisper_model = None
                self._vad_model = vad_model
                self._loaded_model_name = "(remote)"
                self._cleaning_provider = None
                self.active_device_label = "Remote (P620)"
            self._get_cleaning_provider()
            log.info("Remote backend active — local Whisper not loaded")
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
        vad_model = get_vad_model()
        with self._lock:
            self._whisper_model = whisper_model
            self._vad_model = vad_model
            self._loaded_model_name = target_model
            self._cleaning_provider = None
            self.active_device_label = label
        self._get_cleaning_provider()
        log.info(f"Loaded {target_model} on {label}")

    # ── Toggle guard ──────────────────────────────────────────────────────
    def can_toggle(self):
        now = time.time()
        if now - self._last_toggle < 0.4:
            return False
        remote = self.settings.transcription_backend == BACKEND_REMOTE
        with self._lock:
            not_ready = self._vad_model is None if remote else self._whisper_model is None
            if not_ready or self._transcribing:
                return False
        self._last_toggle = now
        return True

    # ── Recording ─────────────────────────────────────────────────────────
    def start_recording(self, device_id=None, on_vad_stop=None, vad_autostop=True,
                        on_partial=None):
        with self._lock:
            if self._recording:
                return
            self._recording = True
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
                try:
                    stream = sd.InputStream(
                        samplerate=rate, channels=1, dtype="float32",
                        blocksize=int(512 * rate / SAMPLE_RATE),
                        device=dev, callback=cb)
                    self._native_rate = rate  # set before start(): callbacks fire after
                    self._stream = stream
                    stream.start()
                    log.info(f"Recording (dev={dev}, rate={rate})")
                    break
                except Exception as e:
                    last_err = e
                    log.warning(f"Mic open failed (dev={dev}, rate={rate}): {e}")
            else:
                raise last_err or RuntimeError("No working audio input device")
        except Exception as e:
            log.error(f"Mic failed: {e}")
            with self._lock:
                self._recording = False
            raise

        self.play_start_sound()
        if self._vad_model is not None:
            self._vad_worker_thread = threading.Thread(target=self._run_vad_worker, daemon=True)
            self._vad_worker_thread.start()
        if self._on_partial is not None and getattr(self.settings, "live_preview", False):
            self._preview_thread = threading.Thread(target=self._run_preview_worker, daemon=True)
            self._preview_thread.start()

    def _run_preview_worker(self):
        """Live preview: periodically transcribe the audio-so-far and emit a
        partial via on_partial. Best-effort — serialized with the final result by
        _infer_lock, and never blocks or alters it."""
        interval = max(0.6, float(getattr(self.settings, "preview_interval", 1.5)))
        last_len = 0
        while self._recording:
            time.sleep(interval)
            if not self._recording:
                break
            chunks = list(self._audio_chunks)
            if not chunks:
                continue
            try:
                audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
                if len(audio) < SAMPLE_RATE * 0.6 or len(audio) == last_len:
                    continue
                last_len = len(audio)
                if self._native_rate != SAMPLE_RATE:
                    audio = self._resample_linear(audio, self._native_rate, SAMPLE_RATE)
                if self.settings.transcription_backend == BACKEND_REMOTE:
                    text = self._transcribe_remote(audio)
                else:
                    text = self._transcribe_local(audio)
                if text and self._recording and self._on_partial:
                    self._on_partial(text)
            except Exception as e:
                log.debug(f"preview worker: {e}")

    def _stop_vad_worker(self):
        """Signal the VAD worker to drain its queue and exit, then join it."""
        if self._vad_queue is not None:
            try:
                self._vad_queue.put_nowait(None)  # sentinel
            except queue.Full:
                # Queue full: clear one slot so the sentinel lands and the worker exits.
                try:
                    self._vad_queue.get_nowait()
                    self._vad_queue.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
        if self._vad_worker_thread is not None:
            self._vad_worker_thread.join(timeout=2.0)
            self._vad_worker_thread = None

    def stop_recording(self):
        with self._lock:
            if not self._recording:
                return None, 0.0
            self._recording = False
        # Drain VAD before reading _speech_detected so a late chunk isn't lost.
        self._stop_vad_worker()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2.0)  # _infer_lock guards the rest
            self._preview_thread = None
        duration = time.time() - self._record_start
        self.play_stop_sound()
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
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

    def _run_vad_worker(self):
        """VAD inference loop — runs on a dedicated background thread, not the audio callback."""
        while True:
            chunk = self._vad_queue.get()
            if chunk is None:  # sentinel — drain complete, exit cleanly
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
                log.info(f"Dictionary applied: {raw[:60]!r} -> {corrected[:60]!r}")
            log.info(f"Raw: {corrected[:100]!r}")
            return corrected
        except Exception as e:
            log.error(f"Transcribe: {e}")
            return ""
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
        wav = _audio_to_wav_bytes(audio, SAMPLE_RATE)
        url = self.settings.remote_url.rstrip("/") + "/audio/transcriptions"
        try:
            with self._infer_lock:  # final POST waits for any in-flight preview POST
                body = _post_multipart_audio(url, wav, fields, timeout=60.0)
            return (body.get("text") or "").strip()
        except Exception as e:
            log.error(f"Remote ASR failed ({self.settings.remote_url}): {e}")
            return ""

    # ── AI cleaning ───────────────────────────────────────────────────────
    def clean_text(self, text):
        # Raw mode: skip the LLM entirely and keep the exact transcribed words.
        if self.settings.cleaning_preset == RAW_PRESET:
            return CleanResult(text=text)
        try:
            provider = self._get_cleaning_provider()
            if provider is None:
                log.warning("No cleaning provider configured — returning raw")
                return CleanResult(text=text, used_fallback=True, reason="provider unavailable")

            log.info(f"Cleaning [{self.settings.clean_provider}/{self.settings.clean_model}]: {text[:120]!r}")
            word_count = len(text.split())
            max_tokens = max(1024, int(word_count * 2.5))
            result = provider.clean(text, max_tokens=max_tokens)
            # Sanity check: cleaned text should be at least 40% of raw length
            if len(result) < len(text) * 0.4 and len(text) > 100:
                log.warning(f"Clean too short ({len(result)} vs {len(text)} raw chars) — using raw")
                return CleanResult(text=text, used_fallback=True, reason="cleaned output too short")
            log.info(f"Clean: {result[:120]!r}")
            return CleanResult(text=result)
        except Exception as e:
            log.error(f"Cleaning API: {e}")
            self._cleaning_provider = None
            return CleanResult(text=text, used_fallback=True, reason=str(e))

    # ── VAD ────────────────────────────────────────────────────────────────
    def _check_vad(self, chunk):
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
            if p >= self.settings.vad_threshold:
                self._speech_detected = True
                self._silence_start = None
            elif p < self.settings.vad_neg_thresh and self._speech_detected:
                if self._silence_start is None:
                    self._silence_start = time.time()
                elif time.time() - self._silence_start >= self.settings.silence_timeout:
                    if self._vad_autostop:  # disabled in push-to-talk
                        self._fire_stop()  # one-shot guarded — safe to reach repeatedly
                    return

    # ── Cleaning provider cache ──────────────────────────────────────────
    def _get_cleaning_provider(self):
        if self.settings.cleaning_preset == RAW_PRESET:
            return None  # Raw mode never cleans
        if self._cleaning_provider is None:
            try:
                provider_name = self.settings.clean_provider
                api_key = self.settings.get_api_key().strip()
                needs_key = PROVIDERS.get(provider_name, {}).get("needs_key", True)
                if needs_key and not api_key:
                    return None
                preset = CLEANING_PRESETS.get(
                    self.settings.cleaning_preset, CLEANING_PRESETS["General"])
                # Dictionary Layer C: tell the LLM these spellings are intentional.
                preset = preset + self.dictionary.prompt_block()
                self._cleaning_provider = create_provider(
                    provider_name, api_key, self.settings.clean_model, preset,
                    endpoint=self.settings.ollama_endpoint if provider_name == "Ollama" else None,
                    timeout=15.0)
            except Exception as e:
                log.error(f"Provider init [{self.settings.clean_provider}]: {e}")
                return None
        return self._cleaning_provider

    def invalidate_provider(self):
        self._cleaning_provider = None

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
    """Always-on-top pill at bottom-center of screen. Shows during recording/transcribing."""

    def __init__(self, root):
        self._root = root
        self._win = None
        self._dot = self._label = self._timer = None
        self._pulse_job = None
        self._dot_on = True

    def show(self, state="recording"):
        if self._win is None:
            self._create()
        if state == "recording":
            self._dot.configure(text_color="#ef5350")
            self._label.configure(text="Recording")
            self._timer.configure(text="00:00.0")
            self._timer.pack(side="right", padx=(0, 14))
            self._start_pulse()
        elif state == "transcribing":
            self._stop_pulse()
            self._dot.configure(text_color="#ff9800")
            self._label.configure(text="Transcribing...")
            self._timer.pack_forget()
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

    def _get_current_monitor(self):
        """Get the geometry of the monitor the main window is on."""
        try:
            result = subprocess.run(
                ["xrandr", "--listmonitors"],
                capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                raise RuntimeError("xrandr failed")
            # Parse monitor lines like: " 0: +*DP-0 5120/700x2880/400+0+0"
            import re
            monitors = []
            for line in result.stdout.strip().split("\n")[1:]:
                m = re.search(r'(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)', line)
                if m:
                    monitors.append((int(m.group(1)), int(m.group(2)),
                                     int(m.group(3)), int(m.group(4))))
            # Find which monitor contains the main window
            wx = self._root.winfo_rootx()
            wy = self._root.winfo_rooty()
            for mw, mh, mx, my in monitors:
                if mx <= wx < mx + mw and my <= wy < my + mh:
                    return mw, mh, mx, my
            # Fallback: primary (first) monitor
            if monitors:
                return monitors[0]
        except Exception:
            pass
        # Final fallback
        return self._root.winfo_screenwidth(), self._root.winfo_screenheight(), 0, 0

    def _create(self):
        self._win = ctk.CTkToplevel(self._root)
        self._win.overrideredirect(True)
        self._win.attributes('-topmost', True)
        self._win.attributes('-alpha', 0.9)
        self._win.configure(fg_color="#0F0F0F")
        # Scale overlay geometry to match widget scaling
        scale = ctk._get_window_scaling(self._win) if hasattr(ctk, '_get_window_scaling') else 1.0
        try:
            scale = self._win._get_widget_scaling()
        except Exception:
            scale = detect_ui_scale()
        w, h = int(240 * scale), int(44 * scale)
        mon_w, mon_h, mon_x, mon_y = self._get_current_monitor()
        x = mon_x + (mon_w - w) // 2
        y = mon_y + mon_h - h - 80
        self._win.geometry(f"{w}x{h}+{x}+{y}")
        pill = ctk.CTkFrame(self._win, fg_color="#191919", corner_radius=22)
        pill.pack(fill="both", expand=True, padx=2, pady=2)
        self._dot = ctk.CTkLabel(pill, text="\u25cf", font=("Inter", 18),
                                 text_color="#ef5350", width=20)
        self._dot.pack(side="left", padx=(14, 6))
        self._label = ctk.CTkLabel(pill, text="Recording",
                                   font=("Inter Tight", 13, "bold"),
                                   text_color="#e4e4e7")
        self._label.pack(side="left")
        self._timer = ctk.CTkLabel(pill, text="00:00.0",
                                   font=("Inter", 12), text_color="#71717a")
        self._timer.pack(side="right", padx=(0, 14))
        self._win.withdraw()

    def _start_pulse(self):
        self._dot_on = True
        self._pulse()

    def _stop_pulse(self):
        if self._pulse_job:
            self._root.after_cancel(self._pulse_job)
            self._pulse_job = None
        self._dot_on = True
        if self._dot:
            self._dot.configure(text_color="#ef5350")

    def _pulse(self):
        self._dot_on = not self._dot_on
        self._dot.configure(text_color="#ef5350" if self._dot_on else "#333333")
        self._pulse_job = self._root.after(500, self._pulse)


# ═══════════════════════════════════════════════════════════════════════════════
# APP — GUI + Tray + Hotkey + Orchestration
# ═══════════════════════════════════════════════════════════════════════════════
class ZenVoxApp:
    # ZenVox palette
    BG     = "#0F0F0F"
    PANEL  = "#191919"
    TEXT   = "#e4e4e7"
    MUTED  = "#71717a"
    TEAL   = "#4ECDB8"
    TEAL_H = "#3db8a3"
    BORDER = "#27272a"

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
        self._state = AppState.LOADING
        self._state_since = time.time()
        self._stopping = False
        self._load_generation = 0
        self._pending_loads = 0
        self._last_clean_reason = ""
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

    def _pal(self):
        return (self.BG, self.PANEL, self.TEXT, self.MUTED,
                self.TEAL, self.TEAL_H, self.BORDER)

    def _build_gui(self):
        B, P, T, M, TL, TH, BD = self._pal()
        self.root.configure(fg_color=B)
        self.gui_status = ctk.StringVar(value="Loading...")
        self._views = {}
        self._nav_buttons = {}

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ─── Sidebar nav ───
        sidebar = ctk.CTkFrame(self.root, fg_color=P, corner_radius=0, width=210)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

        # ─── Content area ───
        content = ctk.CTkFrame(self.root, fg_color=B, corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(content, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(20, 6))
        self._view_title = ctk.CTkLabel(header, text="Dictate",
                                        font=("Inter Tight", 20, "bold"), text_color=T)
        self._view_title.pack(side="left")
        self._status_dot = ctk.CTkLabel(header, text="●", font=("Inter", 13),
                                        text_color=self.MUTED, width=16)
        self._status_dot.pack(side="right", padx=(8, 0))
        ctk.CTkLabel(header, textvariable=self.gui_status, font=("Inter", 13),
                     text_color="#a1a1aa").pack(side="right")

        container = ctk.CTkFrame(content, fg_color="transparent")
        container.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 20))
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
        B, P, T, M, TL, TH, BD = self._pal()
        wm = ctk.CTkFrame(parent, fg_color="transparent")
        wm.pack(fill="x", padx=22, pady=(24, 22))
        ctk.CTkLabel(wm, text="Zen", font=("Inter Tight", 22, "bold"),
                     text_color=T).pack(side="left")
        ctk.CTkLabel(wm, text="Vox", font=("Inter Tight", 22, "bold"),
                     text_color=TL).pack(side="left")
        for name, label in self.NAV_ITEMS:
            btn = ctk.CTkButton(
                parent, text=label, anchor="w", fg_color="transparent",
                hover_color=BD, text_color=M, font=("Inter Tight", 15),
                corner_radius=10, height=42, command=lambda n=name: self._show_view(n))
            btn.pack(fill="x", padx=12, pady=2)
            self._nav_buttons[name] = btn
        ctk.CTkFrame(parent, fg_color="transparent").pack(fill="both", expand=True)
        ctk.CTkButton(parent, text="Hide to tray", command=self._hide_window,
                      fg_color=BD, hover_color="#333333", text_color=T,
                      font=("Inter Tight", 12), corner_radius=8, height=34
                      ).pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkButton(parent, text="Quit", command=self._quit_from_ui,
                      fg_color="#7f1d1d", hover_color="#991b1b", text_color="#ffffff",
                      font=("Inter Tight", 12), corner_radius=8, height=34
                      ).pack(fill="x", padx=12, pady=(0, 16))

    def _show_view(self, name):
        view = self._views.get(name)
        if view is None:
            return
        view.tkraise()
        self._view_title.configure(text=dict(self.NAV_ITEMS).get(name, name))
        for n, btn in self._nav_buttons.items():
            active = (n == name)
            btn.configure(text_color=self.TEAL if active else self.MUTED,
                          fg_color=self.BORDER if active else "transparent")
        if name == "dictionary":
            self._refresh_dictionary()
        elif name == "history":
            self._refresh_history()

    # ── View: Dictate (home) ──────────────────────────────────────────────
    def _build_view_home(self, parent):
        B, P, T, M, TL, TH, BD = self._pal()
        card = ctk.CTkFrame(parent, fg_color=P, border_color=BD, border_width=1,
                            corner_radius=16)
        card.pack(fill="x", pady=(0, 14))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=18)
        self._rec_btn = ctk.CTkButton(
            inner, text="●", width=56, height=56, corner_radius=28,
            fg_color="#ef5350", hover_color="#c62828", text_color="#fff",
            font=("Inter", 26), command=self._gui_toggle_record)
        self._rec_btn.pack(side="left", padx=(0, 18))
        meta = ctk.CTkFrame(inner, fg_color="transparent")
        meta.pack(side="left", fill="x", expand=True)
        self._timer_label = ctk.CTkLabel(meta, text="Ready", anchor="w",
                                         font=("Inter Tight", 15, "bold"), text_color=M)
        self._timer_label.pack(fill="x")
        self._level_bar = ctk.CTkProgressBar(meta, progress_color="#ef5350",
                                            fg_color=B, height=8, corner_radius=4)
        self._level_bar.pack(fill="x", pady=(8, 0))
        self._level_bar.set(0)
        hk = self.settings.hotkey_record.lower()
        ctk.CTkLabel(meta, text=f"Press  {hk}  to dictate into any app",
                     font=("Inter", 11), text_color=M, anchor="w").pack(fill="x", pady=(6, 0))

        ctk.CTkLabel(parent, text="Last transcription", font=("Inter", 13, "bold"),
                     text_color=M).pack(anchor="w", pady=(4, 4))
        self.gui_text = ctk.CTkTextbox(parent, wrap="word", state="disabled",
                                       fg_color=P, text_color=T, font=("Inter", 14),
                                       border_color=BD, border_width=1, corner_radius=12)
        self.gui_text.pack(fill="both", expand=True)
        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(bf, text="Re-paste", command=self._gui_repaste,
                      fg_color=BD, hover_color=TL, text_color=T,
                      font=("Inter Tight", 13, "bold"), corner_radius=8,
                      width=110, height=38).pack(side="right", padx=(8, 0))
        ctk.CTkButton(bf, text="Copy", command=self._gui_copy,
                      fg_color=TL, hover_color=TH, text_color=B,
                      font=("Inter Tight", 13, "bold"), corner_radius=8,
                      width=110, height=38).pack(side="right")

    # ── View: History ─────────────────────────────────────────────────────
    def _build_view_history(self, parent):
        B, P, T, M, TL, TH, BD = self._pal()
        self._search_var = ctk.StringVar()
        se = ctk.CTkEntry(parent, textvariable=self._search_var,
                          placeholder_text="Search history...", height=38,
                          fg_color=P, border_color=BD, text_color=T, font=("Inter", 13))
        se.pack(fill="x", pady=(0, 10))
        self._search_var.trace_add("write", lambda *a: self._refresh_history())
        self._hist_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._hist_frame.pack(fill="both", expand=True)
        hbf = ctk.CTkFrame(parent, fg_color="transparent")
        hbf.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(hbf, text="Clear History", fg_color=BD, hover_color="#ef5350",
                      text_color=T, font=("Inter", 12), corner_radius=8,
                      width=120, height=32, command=self._gui_clear_history).pack(side="right")

    # ── View: Dictionary ──────────────────────────────────────────────────
    def _build_view_dictionary(self, parent):
        B, P, T, M, TL, TH, BD = self._pal()
        ctk.CTkLabel(parent, anchor="w", font=("Inter", 12), text_color=M,
                     text="Add words, names, and jargon so they come out spelled exactly right — every time."
                     ).pack(fill="x", pady=(0, 10))
        addcard = ctk.CTkFrame(parent, fg_color=P, border_color=BD, border_width=1,
                              corner_radius=12)
        addcard.pack(fill="x", pady=(0, 12))
        row = ctk.CTkFrame(addcard, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=12)
        self._dict_written = ctk.CTkEntry(row, placeholder_text="Written (e.g. ZenVox)",
                                          width=190, fg_color=B, border_color=BD,
                                          text_color=T, font=("Inter", 12))
        self._dict_written.pack(side="left", padx=(0, 8))
        self._dict_spoken = ctk.CTkEntry(row, placeholder_text="Sounds like (comma-separated, optional)",
                                         fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))
        self._dict_spoken.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._dict_boost = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(row, variable=self._dict_boost, text="Boost only",
                        fg_color=TL, hover_color=TH, text_color=M, font=("Inter", 11),
                        checkbox_width=18, checkbox_height=18).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row, text="Add", command=self._dict_add, fg_color=TL,
                      hover_color=TH, text_color=B, font=("Inter Tight", 12, "bold"),
                      corner_radius=8, width=70, height=32).pack(side="left")
        self._dict_written.bind("<Return>", lambda e: self._dict_add())
        self._dict_spoken.bind("<Return>", lambda e: self._dict_add())
        self._dict_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._dict_frame.pack(fill="both", expand=True)

    def _refresh_dictionary(self):
        if not hasattr(self, "_dict_frame"):
            return
        B, P, T, M, TL, TH, BD = self._pal()
        for w in self._dict_frame.winfo_children():
            w.destroy()
        entries = sorted(self.engine.dictionary.entries, key=lambda e: e.written.lower())
        if not entries:
            ctk.CTkLabel(self._dict_frame, font=("Inter", 12), text_color=M,
                         text="No words yet. Add names or jargon above.").pack(anchor="w", padx=8, pady=12)
            return
        for e in entries:
            f = ctk.CTkFrame(self._dict_frame, fg_color=P, corner_radius=8)
            f.pack(fill="x", pady=3)
            inner = ctk.CTkFrame(f, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=8)
            ctk.CTkLabel(inner, text=e.written, font=("Inter Tight", 13, "bold"),
                         text_color=TL).pack(side="left")
            if e.boost_only:
                ctk.CTkLabel(inner, text="  boost", font=("Inter", 10),
                             text_color=M).pack(side="left")
            if e.spoken:
                ctk.CTkLabel(inner, text="  ← " + ", ".join(e.spoken),
                             font=("Inter", 11), text_color=M, anchor="w"
                             ).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(inner, text="✕", width=28, height=24, fg_color=BD,
                          hover_color="#ef5350", text_color=T, font=("Inter", 12),
                          corner_radius=6,
                          command=lambda w=e.written: self._dict_delete(w)).pack(side="right")

    def _dict_add(self):
        written = self._dict_written.get().strip()
        if not written:
            return
        spoken = [s.strip() for s in self._dict_spoken.get().split(",") if s.strip()]
        self.engine.dictionary.add(DictionaryEntry(
            written=written, spoken=spoken, boost_only=bool(self._dict_boost.get())))
        self.engine.invalidate_provider()  # Layer C must pick up new vocab
        self._dict_written.delete(0, "end")
        self._dict_spoken.delete(0, "end")
        self._dict_boost.set(False)
        self._refresh_dictionary()

    def _dict_delete(self, written):
        self.engine.dictionary.delete(written)
        self.engine.invalidate_provider()
        self._refresh_dictionary()

    # ── View: Settings ────────────────────────────────────────────────────
    def _build_view_settings(self, parent):
        B, P, T, M, TL, TH, BD = self._pal()
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        combo = dict(fg_color=B, border_color=BD, button_color=BD, button_hover_color=TL,
                     dropdown_fg_color=P, dropdown_hover_color=TL, font=("Inter", 12), text_color=T)
        entry = dict(fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))

        def section(title):
            ctk.CTkLabel(scroll, text=title, font=("Inter Tight", 14, "bold"),
                         text_color=T, anchor="w").pack(fill="x", pady=(14, 6))
            card = ctk.CTkFrame(scroll, fg_color=P, border_color=BD, border_width=1, corner_radius=12)
            card.pack(fill="x")
            return card

        # Transcription
        c = section("Transcription")
        r = ctk.CTkFrame(c, fg_color="transparent")
        r.pack(fill="x", padx=16, pady=14)
        self.gui_model = ctk.StringVar(value=self.settings.model_name)
        ctk.CTkComboBox(r, variable=self.gui_model, values=MODELS, width=180,
                        command=self._on_model, **combo).pack(side="left", padx=(0, 8))
        self.gui_lang = ctk.StringVar(value=self.settings.lang_name)
        ctk.CTkComboBox(r, variable=self.gui_lang, values=list(LANGS.keys()), width=150,
                        command=self._on_lang, **combo).pack(side="left", padx=(0, 8))
        self.gui_mic = ctk.StringVar(value=self.settings.mic_name)
        ctk.CTkComboBox(r, variable=self.gui_mic, values=[n for _, n in self.input_devs],
                        command=self._on_mic, **combo).pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(c, anchor="w", font=("Inter", 11), text_color=M,
                     text="Tip: keep Language on Auto-detect for franglais — forcing French mistranscribes English."
                     ).pack(fill="x", padx=16, pady=(0, 8))
        rb = ctk.CTkFrame(c, fg_color="transparent")
        rb.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkLabel(rb, text="Run on:", font=("Inter", 12), text_color=M).pack(side="left", padx=(0, 4))
        self._backend_labels = {BACKEND_LOCAL: "This machine (local GPU/CPU)",
                                BACKEND_REMOTE: "Remote server (P620)"}
        self.gui_backend = ctk.StringVar(
            value=self._backend_labels.get(self.settings.transcription_backend, self._backend_labels[BACKEND_LOCAL]))
        ctk.CTkComboBox(rb, variable=self.gui_backend, values=list(self._backend_labels.values()),
                        width=230, command=self._on_backend, **combo).pack(side="left", padx=(0, 8))
        self.gui_remote_url = ctk.StringVar(value=self.settings.remote_url)
        rue = ctk.CTkEntry(rb, textvariable=self.gui_remote_url, placeholder_text="http://host:8771/v1", **entry)
        rue.pack(side="left", fill="x", expand=True)
        rue.bind("<FocusOut>", lambda e: self._on_remote_url())

        # AI Cleaning — provider + key + model MUST share one frame
        # (_sync_provider_inputs re-packs the key entry with before=self._model_entry).
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
        ctk.CTkLabel(r2, text="Cleaning style:", font=("Inter", 12),
                     text_color=M).pack(side="left", padx=(0, 8))
        self.gui_preset = ctk.StringVar(value=self.settings.cleaning_preset)
        ctk.CTkComboBox(r2, variable=self.gui_preset, values=PRESET_NAMES,
                        width=150, command=self._on_preset, **combo).pack(side="left")
        ctk.CTkLabel(r2, text="(Raw = no AI cleanup, exact words)", font=("Inter", 11),
                     text_color=M).pack(side="left", padx=(10, 0))

        # Behavior
        c = section("Behavior")
        r = ctk.CTkFrame(c, fg_color="transparent")
        r.pack(fill="x", padx=16, pady=14)
        ctk.CTkLabel(r, text="Silence stop:", font=("Inter", 12),
                     text_color=M).pack(side="left", padx=(0, 4))
        self.gui_silence = ctk.StringVar(value=str(self.settings.silence_timeout))
        sle = ctk.CTkEntry(r, textvariable=self.gui_silence, width=56, **entry)
        sle.pack(side="left")
        sle.bind("<FocusOut>", lambda e: self._on_silence())
        ctk.CTkLabel(r, text="s", font=("Inter", 11), text_color=M).pack(side="left", padx=(2, 16))
        ctk.CTkLabel(r, text="Output:", font=("Inter", 12), text_color=M).pack(side="left", padx=(0, 4))
        self.gui_output = ctk.StringVar(value=self.settings.output_mode)
        ctk.CTkComboBox(r, variable=self.gui_output, values=OUTPUT_MODES, width=150,
                        command=self._on_output, **combo).pack(side="left", padx=(0, 16))
        self.gui_audio = ctk.BooleanVar(value=self.settings.audio_feedback)
        ctk.CTkCheckBox(r, variable=self.gui_audio, text="Sound", fg_color=TL,
                        hover_color=TH, text_color=M, font=("Inter", 12), checkbox_width=18,
                        checkbox_height=18, command=self._on_audio).pack(side="left")
        rb = ctk.CTkFrame(c, fg_color="transparent")
        rb.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkLabel(rb, text="Capture:", font=("Inter", 12), text_color=M).pack(side="left", padx=(0, 4))
        self._capture_labels = {CAPTURE_TOGGLE: "Toggle — press to start, auto-stop on silence",
                                CAPTURE_PTT: "Push-to-talk — hold the record key"}
        self.gui_capture = ctk.StringVar(
            value=self._capture_labels.get(self.settings.capture_mode, self._capture_labels[CAPTURE_TOGGLE]))
        ctk.CTkComboBox(rb, variable=self.gui_capture, values=list(self._capture_labels.values()),
                        width=320, command=self._on_capture_mode, **combo).pack(side="left")
        self.gui_preview = ctk.BooleanVar(value=self.settings.live_preview)
        ctk.CTkCheckBox(rb, variable=self.gui_preview, text="Live preview", fg_color=TL,
                        hover_color=TH, text_color=M, font=("Inter", 12), checkbox_width=18,
                        checkbox_height=18, command=self._on_preview).pack(side="left", padx=(16, 0))

        # Hotkeys
        c = section("Hotkeys")
        hk = ctk.CTkFrame(c, fg_color="transparent")
        hk.pack(fill="x", padx=16, pady=14)
        ctk.CTkLabel(hk, anchor="w", font=("Inter", 12), text_color=T,
                     text=f"{self.settings.hotkey_record.lower()}   →   record / stop").pack(fill="x")
        ctk.CTkLabel(hk, anchor="w", font=("Inter", 12), text_color=T,
                     text=f"{self.settings.hotkey_repaste.lower()}   →   re-paste last").pack(fill="x", pady=(4, 0))
        if not self._tray_has_menu:
            ctk.CTkLabel(c, anchor="w", font=("Inter", 11), text_color=M,
                         text="Tray menus are unavailable on this backend — use this window; Ctrl+Q quits."
                         ).pack(fill="x", padx=16, pady=(0, 12))

    # ── GUI Callbacks ─────────────────────────────────────────────────────
    def _gui_toggle_record(self):
        """Record button click — same as hotkey toggle."""
        self._toggle()

    def _gui_copy(self):
        if self.last_text:
            _copy_text(self.last_text)

    def _gui_repaste(self):
        if self.last_text:
            threading.Thread(target=self._repaste, daemon=True).start()

    def _gui_update_text(self, text):
        self.gui_text.configure(state="normal")
        self.gui_text.delete("1.0", "end")
        self.gui_text.insert("end", text)
        self.gui_text.configure(state="disabled")

    def _on_partial_preview(self, text):
        # Live preview while still recording — show partial text + a cue.
        if self._get_state() == AppState.RECORDING and text:
            self._gui_update_text(text + " …")
            self._overlay.set_label("Listening…")

    def _gui_clear_history(self):
        self.history.clear()
        self._refresh_history()

    def _refresh_history(self):
        for w in self._hist_frame.winfo_children():
            w.destroy()
        q = self._search_var.get().strip()
        entries = self.history.search(q) if q else self.history.get_recent(50)
        B, T, TL, BD = self.BG, self.TEXT, self.TEAL, self.BORDER
        for entry in entries:
            try:
                dt = datetime.fromisoformat(entry.timestamp).astimezone()
                ts = dt.strftime("%H:%M")
            except Exception:
                ts = "??:??"
            preview = entry.cleaned_text[:80]
            if len(entry.cleaned_text) > 80:
                preview += "..."
            dur = f"{entry.duration_sec:.0f}s" if entry.duration_sec else ""
            f = ctk.CTkFrame(self._hist_frame, fg_color=B, corner_radius=8)
            f.pack(fill="x", pady=2)
            inner = ctk.CTkFrame(f, fg_color="transparent")
            inner.pack(fill="x", padx=10, pady=6)
            ctk.CTkLabel(inner, text=ts, font=("Inter", 11, "bold"),
                         text_color=TL, width=40).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(inner, text=preview, font=("Inter", 12),
                         text_color=T, anchor="w").pack(side="left", fill="x", expand=True)
            if dur:
                ctk.CTkLabel(inner, text=dur, font=("Inter", 10),
                             text_color=self.MUTED, width=30).pack(side="right", padx=(4, 4))
            txt = entry.cleaned_text
            ctk.CTkButton(inner, text="Copy", width=50, height=24, fg_color=BD,
                          hover_color=TL, text_color=T, font=("Inter", 11),
                          corner_radius=6,
                          command=lambda t=txt: _copy_text(t)).pack(side="right")

    # ── App State ────────────────────────────────────────────────────────
    def _queue_ui(self, callback):
        try:
            self.root.after(0, callback)
        except RuntimeError:
            pass

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
            self.icon.icon = ICONS.get(icon_state, ICONS["idle"])
            self.icon.title = f"ZenVox - {msg}"
        self._queue_ui(lambda msg=msg: self.gui_status.set(msg))
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
        if (state in (AppState.TRANSCRIBING, AppState.CLEANING)
                and elapsed > STUCK_PIPELINE_TIMEOUT):
            log.error(f"Watchdog: stuck in {state.value} for {elapsed:.0f}s — forcing reset")
            self.engine.force_reset()
            self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}] - recovered from stall")
            self._overlay.hide()
            self._update_rec_bar()
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

    def _on_lang(self, v=None):
        if not self._guard_setting_change(
                "language", revert=lambda: self.gui_lang.set(self.settings.lang_name)):
            return
        self.settings.lang_name = self.gui_lang.get()
        self.settings.save()

    def _on_mic(self, v=None):
        if not self._guard_setting_change(
                "microphone", revert=lambda: self.gui_mic.set(self.settings.mic_name)):
            return
        self.settings.mic_name = self.gui_mic.get()
        self.settings.save()

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
        if new_provider == "Ollama":
            self._key_entry.configure(show="", placeholder_text="http://localhost:11434/v1")
            self.gui_key.set(self.settings.ollama_endpoint)
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
        if self.settings.clean_provider == "Ollama":
            val = self.gui_key.get().strip()
            self.settings.ollama_endpoint = val or "http://localhost:11434/v1"
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

    def _on_silence(self):
        if not self._guard_setting_change(
                "silence timeout", revert=lambda: self.gui_silence.set(str(self.settings.silence_timeout))):
            return
        try:
            v = float(self.gui_silence.get())
            if 0.5 <= v <= 10.0:
                self.settings.silence_timeout = v
                self.settings.save()
            else:
                self.gui_silence.set(str(self.settings.silence_timeout))
        except ValueError:
            self.gui_silence.set(str(self.settings.silence_timeout))

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
        if provider == "Ollama":
            self._key_entry.configure(show="", placeholder_text="http://localhost:11434/v1")
            self.gui_key.set(self.settings.ollama_endpoint)
            self._key_entry.pack(side="left", padx=(0, 8), before=self._model_entry)
        else:
            needs_key = PROVIDERS.get(provider, {}).get("needs_key", True)
            self._key_entry.configure(show="*", placeholder_text="API key")
            self.gui_key.set(self.settings.get_api_key())
            if needs_key:
                self._key_entry.pack(side="left", padx=(0, 8), before=self._model_entry)
            else:
                self._key_entry.pack_forget()

    # ── Status ────────────────────────────────────────────────────────────
    def _set_status(self, state, tooltip=None):
        if isinstance(state, AppState):
            app_state = state
        else:
            try:
                app_state = AppState(state)
            except ValueError:
                app_state = AppState.IDLE
        self._set_app_state(app_state, tooltip)

    # ── Recording bar ─────────────────────────────────────────────────────
    def _update_rec_bar(self):
        state = self._get_state()
        if state == AppState.RECORDING and self.engine.is_recording:
            self._level_bar.set(self.engine.audio_level)
            d = self.engine.recording_duration
            timer_str = f"{int(d // 60):02d}:{d % 60:04.1f}"
            self._timer_label.configure(
                text=f"Recording  {timer_str}",
                text_color="#ef5350")
            # Button shows stop icon during recording
            self._rec_btn.configure(text="\u25a0", fg_color="#c62828",
                                    hover_color="#b71c1c")
            self._overlay.update_timer(timer_str)
            self._timer_job = self.root.after(100, self._update_rec_bar)
        elif state in (AppState.TRANSCRIBING, AppState.CLEANING):
            self._level_bar.set(0)
            label = "Cleaning..." if state == AppState.CLEANING else "Transcribing..."
            self._timer_label.configure(text=label, text_color="#ff9800")
            self._rec_btn.configure(text="\u25cf", fg_color=self.BORDER,
                                    hover_color=self.BORDER, state="disabled")
        elif state == AppState.LOADING:
            self._level_bar.set(0)
            self._timer_label.configure(text="Loading model...",
                                        text_color=self.MUTED)
            self._rec_btn.configure(text="\u25cf", fg_color=self.BORDER,
                                    hover_color=self.BORDER, state="disabled")
        elif state == AppState.ERROR:
            self._level_bar.set(0)
            self._timer_label.configure(text="Model load failed",
                                        text_color="#ef5350")
            self._rec_btn.configure(text="\u25cf", fg_color=self.BORDER,
                                    hover_color=self.BORDER, state="disabled")
        else:
            self._level_bar.set(0)
            self._timer_label.configure(text="Ready", text_color=self.MUTED)
            self._rec_btn.configure(text="\u25cf", fg_color="#ef5350",
                                    hover_color="#c62828", state="normal")

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
        if self.last_text:
            _copy_text(self.last_text)

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
        self.history.close()
        self._load_executor.shutdown(wait=False, cancel_futures=True)
        self._pipeline_executor.shutdown(wait=False, cancel_futures=True)
        _cleanup_clipboard_owners(list(_PERSISTENT_SELECTION_OWNERS.values()))
        _PERSISTENT_SELECTION_OWNERS.clear()
        self.icon.stop()
        self.root.after(0, self.root.quit)

    # ── Model Loading ─────────────────────────────────────────────────────
    def _load_model(self):
        self._schedule_model_load(self.settings.model_name)

    def _run_hotkey_listener(self):
        try:
            self._hotkey_listener()
        except Exception:
            log.exception("Hotkey listener stopped")

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
        parts = [p.strip().lower() for p in hotkey_str.split("+")]
        keys = set()
        for p in parts:
            if p in MODIFIER_MAP:
                keys.add(MODIFIER_MAP[p])
            elif p in KEY_MAP:
                keys.add(KEY_MAP[p])
            elif len(p) == 1 and p.isalnum():
                keys.add(KeyCode.from_char(p))
        return frozenset(keys)

    def _hotkey_listener(self):
        from pynput.keyboard import Listener

        rec_combo = self._parse_hotkey_pynput(self.settings.hotkey_record)
        rep_combo = self._parse_hotkey_pynput(self.settings.hotkey_repaste)
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

        def on_press(key):
            # Runs on pynput's listener thread. An uncaught exception here KILLS
            # the listener — the tray stays up but the record hotkey goes dead.
            # So: never do real work here, never raise. Work runs on its own
            # short-lived thread (off both the listener and the Tk main loop).
            try:
                current_keys.add(key)
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
        if not self.engine.can_toggle():
            return
        if not self.engine.is_recording:
            self._start_recording()
        else:
            self._stop_and_transcribe()

    def _repaste(self):
        if self.last_text:
            target = _get_active_window_info()
            owners = _copy_text(
                self.last_text,
                include_primary=sys.platform.startswith("linux"),
                for_immediate_paste=True,
            )
            try:
                time.sleep(0.05 if sys.platform == "win32" else 0.35)
                _simulate_paste(window=target)
                time.sleep(0.2 if sys.platform == "win32" else 0.6)
                log.info("Re-pasted")
            finally:
                _cleanup_clipboard_owners(owners)

    # ── Recording Flow ────────────────────────────────────────────────────
    def _start_recording(self):
        # Capture the active window BEFORE overlay steals focus
        self._paste_target_window = _get_active_window_info()
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
        try:
            self._pipeline_executor.submit(self._stop_and_transcribe_task)
        except RuntimeError:
            log.exception("Pipeline executor is unavailable")
            self.engine.force_reset()  # release stream + VAD worker even when gone
            self._set_app_state(AppState.IDLE, "Pipeline unavailable - try again")
            self.root.after(0, self._overlay.hide)

    def _stop_and_transcribe_task(self):
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
            self.root.after(0, self._update_rec_bar)
            if audio is None:
                self._set_app_state(AppState.IDLE, f"Ready [{self.engine.active_device_label}]")
                self.root.after(0, self._overlay.hide)
                return
            self._last_clean_reason = ""
            self._set_app_state(AppState.TRANSCRIBING, "Transcribing...")
            self.root.after(0, lambda: self._overlay.show("transcribing"))
            self.root.after(0, self._update_rec_bar)
            self._transcribe(audio, duration)
        finally:
            self._stopping = False

    def _transcribe(self, audio, duration):
        try:
            raw = self.engine.transcribe(
                audio,
                on_segment=lambda t: self.root.after(
                    0, lambda t=t: self._gui_update_text(t)))
            if not raw:
                return
            self._set_app_state(
                AppState.CLEANING,
                f"Cleaning [{self.settings.clean_model}]...")
            self.root.after(0, lambda: self._overlay.set_label("Cleaning..."))
            clean_result = self.engine.clean_text(raw)
            if clean_result.used_fallback and clean_result.reason:
                self._last_clean_reason = clean_result.reason
                log.warning(f"Using raw transcription fallback: {clean_result.reason}")
            else:
                self._last_clean_reason = ""
            text = clean_result.text
            self.last_text = text
            self.root.after(0, lambda t=text: self._gui_update_text(t))
            self.icon.menu = self._build_menu()

            # History
            self.history.add(
                raw_text=raw, cleaned_text=text,
                language=self.settings.lang_name,
                duration_sec=round(duration, 1),
                model=self.settings.model_name,
                cleaning_preset=self.settings.cleaning_preset)
            self.root.after(0, self._refresh_history)

            # Output
            self._output_text(text)
        except Exception as e:
            log.exception(f"Pipeline: {e}")
            self.last_text = f"[Error: {e}]"
            self.root.after(0, lambda: self._gui_update_text(self.last_text))
        finally:
            ready_msg = f"Ready [{self.engine.active_device_label}]"
            if self._last_clean_reason:
                ready_msg = f"Ready [{self.engine.active_device_label}] - raw text kept ({self._last_clean_reason})"
            self._set_app_state(AppState.IDLE, ready_msg)
            self.root.after(0, self._overlay.hide)
            self.root.after(0, self._update_rec_bar)

    def _output_text(self, text):
        mode = self.settings.output_mode
        if mode == "Auto-paste":
            prev = _read_clipboard_state()
            owners = _copy_text(
                text,
                include_primary=sys.platform.startswith("linux"),
                for_immediate_paste=True,
            )
            try:
                # Linux clipboard ownership is async; give it a little more time.
                paste_delay = 0.05 if sys.platform == "win32" else 0.35
                time.sleep(paste_delay)
                _simulate_paste(window=self._paste_target_window)
                restore_delay = 0.15 if sys.platform == "win32" else 1.0
                time.sleep(restore_delay)
                _restore_clipboard_state(prev, expected_text=text, last_pasted_text=self._last_pasted)
                self._last_pasted = text
            finally:
                _cleanup_clipboard_owners(owners)
        elif mode == "Clipboard only":
            _copy_text(text)
            log.info("Clipboard mode")
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
                    _copy_text(text)
            else:
                _copy_text(text)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ZenVoxApp()


if __name__ == "__main__":
    main()
