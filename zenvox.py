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
    DEVICE, COMPUTE, DEVICE_LABEL, BEEP_START, BEEP_STOP,
    setup_logging, list_input_devices, APP_DIR,
)
from providers import PROVIDERS, PROVIDER_NAMES, create_provider
from history import History

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
# ENGINE — Recording, transcription, cleaning. Thread-safe.
# ═══════════════════════════════════════════════════════════════════════════════
class ZenVoxEngine:
    def __init__(self, settings):
        self.settings = settings
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

    @property
    def is_recording(self):
        with self._lock:
            return self._recording

    @property
    def is_transcribing(self):
        with self._lock:
            return self._transcribing

    @property
    def model_loaded(self):
        with self._lock:
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
        whisper_model = WhisperModel(
            target_model, device=DEVICE, compute_type=COMPUTE,
            cpu_threads=CPU_THREADS, num_workers=NUM_WORKERS)
        vad_model = get_vad_model()
        with self._lock:
            self._whisper_model = whisper_model
            self._vad_model = vad_model
            self._loaded_model_name = target_model
            self._cleaning_provider = None
        self._get_cleaning_provider()
        log.info(f"Loaded {target_model} on {DEVICE_LABEL}")

    # ── Toggle guard ──────────────────────────────────────────────────────
    def can_toggle(self):
        now = time.time()
        if now - self._last_toggle < 0.4:
            return False
        with self._lock:
            if self._whisper_model is None or self._transcribing:
                return False
        self._last_toggle = now
        return True

    # ── Recording ─────────────────────────────────────────────────────────
    def start_recording(self, device_id=None, on_vad_stop=None):
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
        self._vad_h = np.zeros((1, 1, 128), dtype="float32")
        self._vad_c = np.zeros((1, 1, 128), dtype="float32")
        self._vad_context = np.zeros(64, dtype="float32")
        self.play_start_sound()

        self._vad_queue = queue.Queue(maxsize=32)
        self._vad_worker_thread = None
        if self._vad_model is not None:
            self._vad_worker_thread = threading.Thread(target=self._run_vad_worker, daemon=True)
            self._vad_worker_thread.start()

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

        # Always record at the device's native sample rate to avoid
        # ALSA silently returning empty data when rate is unsupported.
        # Audio is resampled to SAMPLE_RATE (16kHz) in stop_recording().
        try:
            dev_info = sd.query_devices(device_id) if device_id is not None else sd.query_devices(kind='input')
            self._native_rate = int(dev_info["default_samplerate"])
        except Exception:
            self._native_rate = SAMPLE_RATE
        try:
            self._stream = sd.InputStream(
                samplerate=self._native_rate, channels=1, dtype="float32",
                blocksize=int(512 * self._native_rate / SAMPLE_RATE),
                device=device_id, callback=cb)
            self._stream.start()
            log.info(f"Recording (dev={device_id}, rate={self._native_rate})")
        except Exception as e:
            log.error(f"Mic failed: {e}")
            with self._lock:
                self._recording = False
            raise

    def stop_recording(self):
        with self._lock:
            if not self._recording:
                return None, 0.0
            self._recording = False
        if self._vad_queue is not None:
            try:
                self._vad_queue.put_nowait(None)  # sentinel — stop VAD worker
            except queue.Full:
                pass
        if self._vad_worker_thread is not None:
            self._vad_worker_thread.join(timeout=2.0)  # wait for VAD to drain before checking _speech_detected
            self._vad_worker_thread = None
        duration = time.time() - self._record_start
        self.play_stop_sound()
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            log.error(f"Stream close: {e}")
        if not self._speech_detected:
            log.info("No speech")
            return None, duration
        audio = (np.concatenate(self._audio_chunks, axis=0).flatten()
                 if self._audio_chunks else np.array([]))
        # Resample to SAMPLE_RATE if recorded at a different rate
        if self._native_rate != SAMPLE_RATE and len(audio) > 0:
            num_samples = int(len(audio) * SAMPLE_RATE / self._native_rate)
            indices = np.linspace(0, len(audio) - 1, num_samples).astype(np.float32)
            idx_floor = np.floor(indices).astype(int)
            idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
            frac = indices - idx_floor
            audio = (audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac).astype(np.float32)
            log.info(f"Resampled {self._native_rate}->{SAMPLE_RATE} Hz ({num_samples} samples)")
        log.info(f"Recorded {duration:.1f}s, {len(audio)} samples")
        return audio, duration

    def _resample_vad_chunk(self, chunk):
        """Resample a raw audio chunk from native device rate to 16kHz for VAD."""
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if len(chunk) == 0 or self._native_rate == SAMPLE_RATE:
            return chunk
        return np.interp(
            np.linspace(0, len(chunk), int(len(chunk) * SAMPLE_RATE / self._native_rate)),
            np.arange(len(chunk)),
            chunk,
        ).astype(np.float32)

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
            model = self._whisper_model
        try:
            if len(audio) < SAMPLE_RATE * 0.3:
                return ""
            if model is None:
                raise RuntimeError("Whisper model is not loaded")
            lang = LANGS.get(self.settings.lang_name)
            prompt = ("Transcription en fran\u00e7ais canadien."
                      if self.settings.lang_name == "Fran\u00e7ais (CA)" else None)
            segs, _ = model.transcribe(
                audio, language=lang, beam_size=1,
                initial_prompt=prompt, vad_filter=True)
            parts = []
            for s in segs:
                parts.append(s.text)
                if on_segment:
                    on_segment(" ".join(parts).strip())
            raw = " ".join(parts).strip()
            log.info(f"Raw: {raw[:100]!r}")
            return raw
        except Exception as e:
            log.error(f"Transcribe: {e}")
            return ""
        finally:
            with self._lock:
                self._transcribing = False

    # ── AI cleaning ───────────────────────────────────────────────────────
    def clean_text(self, text):
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
        for off in range(0, len(chunk) - 511, 512):
            w = chunk[off:off + 512]
            frame = np.concatenate([self._vad_context, w]).reshape(1, -1).astype("float32")
            self._vad_context = w[-64:].copy()
            probs, self._vad_h, self._vad_c = self._vad_model.session.run(
                None, {"input": frame, "h": self._vad_h, "c": self._vad_c})
            p = float(probs[0])
            if p >= self.settings.vad_threshold:
                self._speech_detected = True
                self._silence_start = None
            elif p < self.settings.vad_neg_thresh and self._speech_detected:
                if self._silence_start is None:
                    self._silence_start = time.time()
                elif time.time() - self._silence_start >= self.settings.silence_timeout:
                    if self._on_vad_stop:
                        self._on_vad_stop()
                    return

    # ── Cleaning provider cache ──────────────────────────────────────────
    def _get_cleaning_provider(self):
        if self._cleaning_provider is None:
            try:
                provider_name = self.settings.clean_provider
                api_key = self.settings.get_api_key().strip()
                needs_key = PROVIDERS.get(provider_name, {}).get("needs_key", True)
                if needs_key and not api_key:
                    return None
                preset = CLEANING_PRESETS.get(
                    self.settings.cleaning_preset, CLEANING_PRESETS["General"])
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
            scale = 2.5
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
        ctk.set_widget_scaling(2.5)
        self.root = ctk.CTk()
        self.root.title("ZenVox - Voice to Text")
        self.root.geometry("1800x1750")
        self.root.resizable(False, False)
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

        if self._is_first_run:
            # Show the settings window on first run so user can configure
            self.root.deiconify()
            self.root.lift()
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
        self.root.mainloop()

    # ── GUI Build ─────────────────────────────────────────────────────────
    def _build_gui(self):
        B, P, T, M, TL, TH, BD = (
            self.BG, self.PANEL, self.TEXT, self.MUTED,
            self.TEAL, self.TEAL_H, self.BORDER)
        self.root.configure(fg_color=B)
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True)

        # ─── Status bar ───
        self.gui_status = ctk.StringVar(value="Loading...")
        sf = ctk.CTkFrame(main, fg_color=P, border_color=BD, border_width=1, corner_radius=16)
        sf.pack(fill="x", padx=24, pady=(16, 8))
        si = ctk.CTkFrame(sf, fg_color="transparent")
        si.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(si, text="Zen", font=("Inter Tight", 16, "bold"),
                     text_color=T).pack(side="left")
        ctk.CTkLabel(si, text="Vox", font=("Inter Tight", 16, "bold"),
                     text_color=TL).pack(side="left")
        ctk.CTkLabel(si, text="  Voice to Text", font=("Inter Tight", 14),
                     text_color=M).pack(side="left")
        if not self._tray_has_menu:
            action_frame = ctk.CTkFrame(si, fg_color="transparent")
            action_frame.pack(side="right")
            ctk.CTkButton(
                action_frame, text="Hide", command=self._hide_window,
                fg_color=BD, hover_color="#333333", text_color=T,
                font=("Inter Tight", 12, "bold"), corner_radius=8, width=78, height=30
            ).pack(side="right", padx=(8, 0))
            ctk.CTkButton(
                action_frame, text="Quit", command=self._quit_from_ui,
                fg_color="#7f1d1d", hover_color="#991b1b", text_color="#ffffff",
                font=("Inter Tight", 12, "bold"), corner_radius=8, width=78, height=30
            ).pack(side="right")
        ctk.CTkLabel(si, textvariable=self.gui_status, font=("Inter", 13),
                     text_color="#a1a1aa").pack(side="right", padx=(0, 12 if not self._tray_has_menu else 0))

        # ─── Tabs ───
        tabs = ctk.CTkTabview(
            main, fg_color=P, border_color=BD, border_width=1, corner_radius=16,
            segmented_button_fg_color=B,
            segmented_button_selected_color=TL,
            segmented_button_selected_hover_color=TH,
            segmented_button_unselected_color=BD,
            segmented_button_unselected_hover_color="#333")
        tabs.pack(fill="both", expand=True, padx=24, pady=(0, 8))
        t_tab = tabs.add("Transcribe")
        h_tab = tabs.add("History")
        tabs.set("Transcribe")

        # -- Transcribe tab --
        ctk.CTkLabel(t_tab, text="Last transcription",
                     font=("Inter", 13, "bold"), text_color=M
                     ).pack(anchor="w", padx=8, pady=(8, 4))
        self.gui_text = ctk.CTkTextbox(
            t_tab, wrap="word", state="disabled",
            fg_color="transparent", text_color=T, font=("Inter", 14))
        self.gui_text.pack(fill="both", expand=True, padx=4, pady=4)
        bf = ctk.CTkFrame(t_tab, fg_color="transparent")
        bf.pack(fill="x", padx=8, pady=(4, 8))
        ctk.CTkButton(bf, text="Re-paste", command=self._gui_repaste,
                      fg_color=BD, hover_color=TL, text_color=T,
                      font=("Inter Tight", 13, "bold"), corner_radius=8,
                      width=100, height=36).pack(side="right", padx=(8, 0))
        ctk.CTkButton(bf, text="Copy", command=self._gui_copy,
                      fg_color=TL, hover_color=TH, text_color=B,
                      font=("Inter Tight", 13, "bold"), corner_radius=8,
                      width=100, height=36).pack(side="right")

        # -- History tab --
        self._search_var = ctk.StringVar()
        se = ctk.CTkEntry(h_tab, textvariable=self._search_var,
                          placeholder_text="Search history...",
                          fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))
        se.pack(fill="x", padx=8, pady=(12, 8))
        self._search_var.trace_add("write", lambda *a: self._refresh_history())
        self._hist_frame = ctk.CTkScrollableFrame(h_tab, fg_color="transparent")
        self._hist_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        hbf = ctk.CTkFrame(h_tab, fg_color="transparent")
        hbf.pack(fill="x", padx=8, pady=(4, 8))
        ctk.CTkButton(hbf, text="Clear History", fg_color=BD,
                      hover_color="#ef5350", text_color=T, font=("Inter", 12),
                      corner_radius=8, width=110, height=30,
                      command=self._gui_clear_history).pack(side="right")

        # ─── Recording bar ───
        rf = ctk.CTkFrame(main, fg_color=P, border_color=BD, border_width=1,
                          corner_radius=12)
        rf.pack(fill="x", padx=24, pady=(0, 8))
        ri = ctk.CTkFrame(rf, fg_color="transparent")
        ri.pack(fill="x", padx=16, pady=10)
        # Record button — prominent, unmistakable primary action
        self._rec_btn = ctk.CTkButton(
            ri, text="\u25cf", width=40, height=40, corner_radius=20,
            fg_color="#ef5350", hover_color="#c62828", text_color="#fff",
            font=("Inter", 20), command=self._gui_toggle_record)
        self._rec_btn.pack(side="left", padx=(0, 12))
        self._level_bar = ctk.CTkProgressBar(
            ri, progress_color="#ef5350", fg_color=B,
            height=8, width=180, corner_radius=4)
        self._level_bar.pack(side="left", padx=(0, 12))
        self._level_bar.set(0)
        self._timer_label = ctk.CTkLabel(ri, text="Ready",
                                         font=("Inter", 12), text_color=M)
        self._timer_label.pack(side="left")

        # ─── Settings ───
        sp = ctk.CTkFrame(main, fg_color=P, corner_radius=16,
                          border_color=BD, border_width=1)
        sp.pack(fill="x", padx=24, pady=(0, 8))

        # Row 1: Model, Language, Mic
        r1 = ctk.CTkFrame(sp, fg_color="transparent")
        r1.pack(fill="x", padx=20, pady=(12, 8))
        self.gui_model = ctk.StringVar(value=self.settings.model_name)
        ctk.CTkComboBox(
            r1, variable=self.gui_model, values=MODELS, width=155,
            fg_color=B, border_color=BD, button_color=BD, button_hover_color=TL,
            dropdown_fg_color=P, dropdown_hover_color=TL,
            font=("Inter", 12), text_color=T,
            command=self._on_model).pack(side="left", padx=(0, 8))
        self.gui_lang = ctk.StringVar(value=self.settings.lang_name)
        ctk.CTkComboBox(
            r1, variable=self.gui_lang, values=list(LANGS.keys()), width=130,
            fg_color=B, border_color=BD, button_color=BD, button_hover_color=TL,
            dropdown_fg_color=P, dropdown_hover_color=TL,
            font=("Inter", 12), text_color=T,
            command=self._on_lang).pack(side="left", padx=(0, 8))
        self.gui_mic = ctk.StringVar(value=self.settings.mic_name)
        ctk.CTkComboBox(
            r1, variable=self.gui_mic, values=[n for _, n in self.input_devs],
            fg_color=B, border_color=BD, button_color=BD, button_hover_color=TL,
            dropdown_fg_color=P, dropdown_hover_color=TL,
            font=("Inter", 12), text_color=T,
            command=self._on_mic).pack(side="left", fill="x", expand=True)

        # Row 2: Provider + API key + Model
        r2 = ctk.CTkFrame(sp, fg_color="transparent")
        r2.pack(fill="x", padx=20, pady=(0, 8))
        ctk.CTkLabel(r2, text="AI:", font=("Inter", 12, "bold"),
                     text_color=M, width=25).pack(side="left", padx=(0, 4))
        self.gui_provider = ctk.StringVar(value=self.settings.clean_provider)
        ctk.CTkComboBox(
            r2, variable=self.gui_provider, values=PROVIDER_NAMES, width=105,
            fg_color=B, border_color=BD, button_color=BD, button_hover_color=TL,
            dropdown_fg_color=P, dropdown_hover_color=TL,
            font=("Inter", 12), text_color=T,
            command=self._on_provider).pack(side="left", padx=(0, 8))
        self.gui_key = ctk.StringVar(value=self.settings.get_api_key())
        self._key_entry = ctk.CTkEntry(r2, textvariable=self.gui_key, show="*",
                          placeholder_text="API key", width=200,
                          fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))
        self._key_entry.pack(side="left", padx=(0, 8))
        self._key_entry.bind("<FocusOut>", lambda e: self._on_key())
        self.gui_clean = ctk.StringVar(value=self.settings.clean_model)
        ce = ctk.CTkEntry(r2, textvariable=self.gui_clean,
                          placeholder_text=PROVIDERS.get(self.settings.clean_provider, {}).get("default_model", ""),
                          width=180, fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))
        ce.pack(side="left", fill="x", expand=True)
        ce.bind("<FocusOut>", lambda e: self._on_clean())
        self._model_entry = ce

        # Row 3: Silence, Output, Cleaning preset
        r3 = ctk.CTkFrame(sp, fg_color="transparent")
        r3.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(r3, text="Silence:", font=("Inter", 12),
                     text_color=M).pack(side="left", padx=(0, 4))
        self.gui_silence = ctk.StringVar(value=str(self.settings.silence_timeout))
        sle = ctk.CTkEntry(r3, textvariable=self.gui_silence, width=50,
                           fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))
        sle.pack(side="left")
        sle.bind("<FocusOut>", lambda e: self._on_silence())
        ctk.CTkLabel(r3, text="s", font=("Inter", 11),
                     text_color=M).pack(side="left", padx=(2, 12))
        ctk.CTkLabel(r3, text="Output:", font=("Inter", 12),
                     text_color=M).pack(side="left", padx=(0, 4))
        self.gui_output = ctk.StringVar(value=self.settings.output_mode)
        ctk.CTkComboBox(
            r3, variable=self.gui_output, values=OUTPUT_MODES, width=135,
            fg_color=B, border_color=BD, button_color=BD, button_hover_color=TL,
            dropdown_fg_color=P, dropdown_hover_color=TL,
            font=("Inter", 12), text_color=T,
            command=self._on_output).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(r3, text="Cleaning:", font=("Inter", 12),
                     text_color=M).pack(side="left", padx=(0, 4))
        self.gui_preset = ctk.StringVar(value=self.settings.cleaning_preset)
        ctk.CTkComboBox(
            r3, variable=self.gui_preset, values=list(CLEANING_PRESETS.keys()),
            width=120, fg_color=B, border_color=BD, button_color=BD,
            button_hover_color=TL, dropdown_fg_color=P, dropdown_hover_color=TL,
            font=("Inter", 12), text_color=T,
            command=self._on_preset).pack(side="left")
        self.gui_audio = ctk.BooleanVar(value=self.settings.audio_feedback)
        ctk.CTkCheckBox(r3, variable=self.gui_audio, text="Sound",
                        fg_color=TL, hover_color=TH, text_color=M,
                        font=("Inter", 12), width=20, checkbox_width=18,
                        checkbox_height=18, command=self._on_audio
                        ).pack(side="right", padx=(8, 0))

        # ─── Footer ───
        ff = ctk.CTkFrame(main, fg_color="transparent")
        ff.pack(fill="x", padx=32, pady=(0, 12))
        hk_rec = self.settings.hotkey_record.lower()
        hk_rep = self.settings.hotkey_repaste.lower()
        ctk.CTkLabel(ff, text=f"{hk_rec} = record  \u00b7  {hk_rep} = re-paste",
                     font=("Inter", 12), text_color="#a1a1aa").pack(side="left")
        if not self._tray_has_menu:
            ctk.CTkLabel(
                ff,
                text=("Tray menus are unavailable on this backend. "
                      "Left-click the tray icon to reopen. Ctrl+Q quits."),
                font=("Inter", 11),
                text_color=self.MUTED,
            ).pack(side="left", padx=(16, 0))

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
            self._state = state
        msg = tooltip or {
            AppState.LOADING: "Loading model...",
            AppState.IDLE: f"Ready [{DEVICE_LABEL}]",
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

        self._set_app_state(AppState.IDLE, f"Ready [{DEVICE_LABEL}]")

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

        def on_press(key):
            current_keys.add(key)
            if rec_combo and rec_combo.issubset(current_keys):
                current_keys.clear()
                self._toggle()
            elif rep_combo and rep_combo.issubset(current_keys):
                current_keys.clear()
                self._repaste()

        def on_release(key):
            current_keys.discard(key)

        log.info(f"Hotkeys: {self.settings.hotkey_record}=record, {self.settings.hotkey_repaste}=re-paste")
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

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
        try:
            self.engine.start_recording(
                device_id=dev_id,
                on_vad_stop=lambda: self.root.after(0, self._stop_and_transcribe))
            self._set_app_state(AppState.RECORDING, "Recording...")
            self.root.after(0, lambda: self._overlay.show("recording"))
            self.root.after(0, self._update_rec_bar)
        except Exception as e:
            log.exception(f"Start failed: {e}")
            self._set_app_state(AppState.IDLE, "Mic error - check device")
            self.root.after(0, self._update_rec_bar)

    def _stop_and_transcribe(self):
        audio, duration = self.engine.stop_recording()
        self.root.after(0, self._update_rec_bar)
        if audio is None:
            self._set_app_state(AppState.IDLE, f"Ready [{DEVICE_LABEL}]")
            self.root.after(0, self._overlay.hide)
            return
        self._last_clean_reason = ""
        self._set_app_state(AppState.TRANSCRIBING, "Transcribing...")
        self.root.after(0, lambda: self._overlay.show("transcribing"))
        self.root.after(0, self._update_rec_bar)
        try:
            self._pipeline_executor.submit(self._transcribe, audio, duration)
        except RuntimeError:
            log.exception("Pipeline executor is unavailable")
            self._set_app_state(AppState.IDLE, "Pipeline unavailable - try again")
            self.root.after(0, self._overlay.hide)

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
            ready_msg = f"Ready [{DEVICE_LABEL}]"
            if self._last_clean_reason:
                ready_msg = f"Ready [{DEVICE_LABEL}] - raw text kept ({self._last_clean_reason})"
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
