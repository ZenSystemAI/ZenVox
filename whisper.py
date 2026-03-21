#!/usr/bin/env python3
"""
whisper.py  —  ZenSystem Whisper: Voice to text, best in class.
Ctrl+Alt+F12 = record (auto-stops on silence)
Ctrl+Alt+F11 = re-paste last transcription
"""
import concurrent.futures
import os
import sys
import threading
import time
import warnings
import winsound

# Suppress pythonw.exe crashes from writing to missing stderr/stdout
if sys.executable.lower().endswith("pythonw.exe"):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

warnings.filterwarnings("ignore", module="google.generativeai")

import customtkinter as ctk
import numpy as np
import pyautogui
import pyperclip
import pystray
import sounddevice as sd
from datetime import datetime

from config import (
    Settings, SAMPLE_RATE, VAD_THRESHOLD, VAD_NEG_THRESH, CPU_THREADS, NUM_WORKERS,
    MODELS, LANGS, CLEANING_PRESETS, OUTPUT_MODES, ICONS,
    DEVICE, COMPUTE, DEVICE_LABEL, BEEP_START, BEEP_STOP,
    setup_logging, list_input_devices, APP_DIR,
)
from history import History

log = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE — Recording, transcription, cleaning. Thread-safe.
# ═══════════════════════════════════════════════════════════════════════════════
class WhisperEngine:
    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._whisper_model = None
        self._vad_model = None
        self._gemini_model = None
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

    @property
    def is_recording(self):
        return self._recording

    @property
    def is_transcribing(self):
        return self._transcribing

    @property
    def model_loaded(self):
        return self._whisper_model is not None

    @property
    def recording_duration(self):
        return time.time() - self._record_start if self._recording else 0.0

    @property
    def audio_level(self):
        return self._audio_level

    # ── Model ─────────────────────────────────────────────────────────────
    def load_model(self, on_status=None):
        from faster_whisper import WhisperModel
        from faster_whisper.vad import get_vad_model
        if on_status:
            on_status("loading", f"Loading {self.settings.model_name}...")
        self._whisper_model = WhisperModel(
            self.settings.model_name, device=DEVICE, compute_type=COMPUTE,
            cpu_threads=CPU_THREADS, num_workers=NUM_WORKERS)
        self._vad_model = get_vad_model()
        self._get_gemini_model()
        log.info(f"Loaded {self.settings.model_name} on {DEVICE_LABEL}")
        if on_status:
            on_status("idle", f"Ready [{DEVICE_LABEL}]")

    def reload_model(self, on_status=None):
        self._whisper_model = None
        threading.Thread(target=self.load_model, args=(on_status,), daemon=True).start()

    # ── Toggle guard ──────────────────────────────────────────────────────
    def can_toggle(self):
        now = time.time()
        if now - self._last_toggle < 0.4:
            return False
        if not self.model_loaded or self._transcribing:
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

        def cb(indata, frames, t, status):
            if status:
                log.warning(f"Audio: {status}")
            if not self._recording:
                return
            self._audio_chunks.append(indata.copy())
            self._audio_level = min(1.0, float(np.abs(indata).max()) * 5)
            if self._vad_model is not None:
                self._check_vad(indata.flatten())

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=512, device=device_id, callback=cb)
            self._stream.start()
            log.info(f"Recording (dev={device_id})")
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
        log.info(f"Recorded {duration:.1f}s, {len(audio)} samples")
        return audio, duration

    # ── Transcription ─────────────────────────────────────────────────────
    def transcribe(self, audio, on_segment=None):
        self._transcribing = True
        try:
            if len(audio) < SAMPLE_RATE * 0.3:
                return ""
            lang = LANGS.get(self.settings.lang_name)
            prompt = ("Transcription en fran\u00e7ais canadien."
                      if self.settings.lang_name == "Fran\u00e7ais (CA)" else None)
            segs, _ = self._whisper_model.transcribe(
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
            self._transcribing = False

    # ── Gemini cleaning ───────────────────────────────────────────────────
    def clean_text(self, text):
        try:
            model = self._get_gemini_model()
            if model is None:
                log.warning("No Gemini key — raw")
                return text

            def _call():
                log.info(f"Gemini [{self.settings.clean_model}]: {text[:80]!r}")
                r = model.generate_content(
                    f"[RAW] {text} [/RAW]",
                    generation_config={"temperature": 0.2})
                return r.text.strip()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                result = ex.submit(_call).result(timeout=10)
            log.info(f"Clean: {result[:80]!r}")
            return result
        except concurrent.futures.TimeoutError:
            log.warning("Gemini timeout")
            return text
        except Exception as e:
            log.error(f"Gemini: {e}")
            self._gemini_model = None
            return text

    # ── VAD ────────────────────────────────────────────────────────────────
    def _check_vad(self, chunk):
        for off in range(0, len(chunk) - 511, 512):
            w = chunk[off:off + 512]
            frame = np.concatenate([self._vad_context, w]).reshape(1, -1).astype("float32")
            self._vad_context = w[-64:].copy()
            probs, self._vad_h, self._vad_c = self._vad_model.session.run(
                None, {"input": frame, "h": self._vad_h, "c": self._vad_c})
            p = float(probs[0])
            if p >= VAD_THRESHOLD:
                self._speech_detected = True
                self._silence_start = None
            elif p < VAD_NEG_THRESH and self._speech_detected:
                if self._silence_start is None:
                    self._silence_start = time.time()
                elif time.time() - self._silence_start >= self.settings.silence_timeout:
                    if self._on_vad_stop:
                        self._on_vad_stop()
                    return

    # ── Gemini model cache ────────────────────────────────────────────────
    def _get_gemini_model(self):
        if self._gemini_model is None:
            try:
                import google.generativeai as genai
                key = self.settings.gemini_api_key.strip()
                if not key:
                    return None
                genai.configure(api_key=key)
                preset = CLEANING_PRESETS.get(
                    self.settings.cleaning_preset, CLEANING_PRESETS["General"])
                self._gemini_model = genai.GenerativeModel(
                    model_name=self.settings.clean_model,
                    system_instruction=preset)
            except Exception as e:
                log.error(f"Gemini init: {e}")
                return None
        return self._gemini_model

    def invalidate_gemini(self):
        self._gemini_model = None

    # ── Audio feedback ────────────────────────────────────────────────────
    def play_start_sound(self):
        if self.settings.audio_feedback:
            threading.Thread(
                target=lambda: winsound.PlaySound(BEEP_START, winsound.SND_MEMORY),
                daemon=True).start()

    def play_stop_sound(self):
        if self.settings.audio_feedback:
            threading.Thread(
                target=lambda: winsound.PlaySound(BEEP_STOP, winsound.SND_MEMORY),
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

    def _create(self):
        self._win = ctk.CTkToplevel(self._root)
        self._win.overrideredirect(True)
        self._win.attributes('-topmost', True)
        self._win.attributes('-alpha', 0.9)
        self._win.configure(fg_color="#0F0F0F")
        w, h = 240, 44
        sx = self._root.winfo_screenwidth()
        sy = self._root.winfo_screenheight()
        self._win.geometry(f"{w}x{h}+{(sx - w) // 2}+{sy - h - 80}")
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
class WhisperApp:
    # ZenSystem palette
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

        self.engine = WhisperEngine(self.settings)
        self.history = History()
        self.last_text = ""
        self._last_pasted = ""
        self._timer_job = None

        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("Whisper \u2014 ZenSystem")
        self.root.geometry("720x700")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)
        ico = APP_DIR / "whisper.ico"
        if ico.exists():
            self.root.iconbitmap(str(ico))

        self._build_gui()
        self._overlay = FloatingOverlay(self.root)
        self._refresh_history()
        self.root.withdraw()

        self.icon = pystray.Icon("whisper", ICONS["loading"],
                                 "Whisper \u2014 Loading...", menu=self._build_menu())

        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._hotkey_listener, daemon=True).start()

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
        ctk.CTkLabel(si, text="System", font=("Inter Tight", 16, "bold"),
                     text_color=TL).pack(side="left")
        ctk.CTkLabel(si, text=" Whisper", font=("Inter Tight", 14),
                     text_color=M).pack(side="left")
        ctk.CTkLabel(si, textvariable=self.gui_status, font=("Inter", 13),
                     text_color=TL).pack(side="right")

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
        self._level_bar = ctk.CTkProgressBar(
            ri, progress_color="#ef5350", fg_color=B,
            height=8, width=200, corner_radius=4)
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
        r1.pack(fill="x", padx=20, pady=(10, 6))
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

        # Row 2: API key + Gemini model
        r2 = ctk.CTkFrame(sp, fg_color="transparent")
        r2.pack(fill="x", padx=20, pady=(0, 6))
        ctk.CTkLabel(r2, text="API key:", font=("Inter", 12),
                     text_color=M).pack(side="left", padx=(0, 6))
        self.gui_key = ctk.StringVar(value=self.settings.gemini_api_key)
        ke = ctk.CTkEntry(r2, textvariable=self.gui_key, show="*",
                          placeholder_text="AIza...", width=200,
                          fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))
        ke.pack(side="left", padx=(0, 12))
        ke.bind("<FocusOut>", lambda e: self._on_key())
        ctk.CTkLabel(r2, text="Model:", font=("Inter", 12),
                     text_color=M).pack(side="left", padx=(0, 6))
        self.gui_clean = ctk.StringVar(value=self.settings.clean_model)
        ce = ctk.CTkEntry(r2, textvariable=self.gui_clean,
                          placeholder_text="gemini-2.5-flash-lite", width=180,
                          fg_color=B, border_color=BD, text_color=T, font=("Inter", 12))
        ce.pack(side="left", fill="x", expand=True)
        ce.bind("<FocusOut>", lambda e: self._on_clean())

        # Row 3: Silence, Output, Cleaning preset
        r3 = ctk.CTkFrame(sp, fg_color="transparent")
        r3.pack(fill="x", padx=20, pady=(0, 10))
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
        ctk.CTkLabel(ff, text="ctrl+alt+f12 = record  \u00b7  ctrl+alt+f11 = re-paste",
                     font=("Inter", 11), text_color=M).pack(side="left")

    # ── GUI Callbacks ─────────────────────────────────────────────────────
    def _gui_copy(self):
        if self.last_text:
            pyperclip.copy(self.last_text)

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
                          command=lambda t=txt: pyperclip.copy(t)).pack(side="right")

    # ── Settings Callbacks ────────────────────────────────────────────────
    def _on_model(self, v=None):
        self.settings.model_name = self.gui_model.get()
        self.settings.save()
        self.engine.reload_model(self._set_status)

    def _on_lang(self, v=None):
        self.settings.lang_name = self.gui_lang.get()
        self.settings.save()

    def _on_mic(self, v=None):
        self.settings.mic_name = self.gui_mic.get()
        self.settings.save()

    def _on_key(self):
        self.settings.gemini_api_key = self.gui_key.get()
        self.engine.invalidate_gemini()
        self.settings.save()

    def _on_clean(self):
        self.settings.clean_model = self.gui_clean.get()
        self.engine.invalidate_gemini()
        self.settings.save()

    def _on_silence(self):
        try:
            v = float(self.gui_silence.get())
            if 0.5 <= v <= 10.0:
                self.settings.silence_timeout = v
                self.settings.save()
        except ValueError:
            pass

    def _on_output(self, v=None):
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
        self.settings.cleaning_preset = self.gui_preset.get()
        self.engine.invalidate_gemini()
        self.settings.save()

    def _on_audio(self):
        self.settings.audio_feedback = self.gui_audio.get()
        self.settings.save()

    # ── Status ────────────────────────────────────────────────────────────
    def _set_status(self, state, tooltip=None):
        msg = tooltip or f"Whisper \u2014 {state}"
        self.icon.icon = ICONS.get(state, ICONS["idle"])
        self.icon.title = f"Whisper \u2014 {msg}"
        self.root.after(0, lambda: self.gui_status.set(msg))

    # ── Recording bar ─────────────────────────────────────────────────────
    def _update_rec_bar(self):
        if self.engine.is_recording:
            self._level_bar.set(self.engine.audio_level)
            d = self.engine.recording_duration
            timer_str = f"{int(d // 60):02d}:{d % 60:04.1f}"
            self._timer_label.configure(
                text=f"Recording  {timer_str}",
                text_color="#ef5350")
            self._overlay.update_timer(timer_str)
            self._timer_job = self.root.after(100, self._update_rec_bar)
        elif self.engine.is_transcribing:
            self._level_bar.set(0)
            self._timer_label.configure(text="Transcribing...",
                                        text_color="#ff9800")
        else:
            self._level_bar.set(0)
            self._timer_label.configure(text="Ready", text_color=self.MUTED)

    # ── Tray Menu ─────────────────────────────────────────────────────────
    def _build_menu(self):
        def model_action(n):
            def a(icon, item):
                self.settings.model_name = n
                self.settings.save()
                self.root.after(0, lambda: self.gui_model.set(n))
                self.engine.reload_model(self._set_status)
            return a

        def lang_action(n):
            def a(icon, item):
                self.settings.lang_name = n
                self.settings.save()
                self.root.after(0, lambda: self.gui_lang.set(n))
            return a

        def mic_action(n):
            def a(icon, item):
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
                              if self.last_text else "Last: \u2014"),
                self._copy_last),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Window", lambda i, it: self._show_window()),
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
            pyperclip.copy(self.last_text)

    def _show_window(self):
        self.root.after(0, lambda: (
            self.root.deiconify(), self.root.lift(), self.root.focus_force()))

    def _quit(self, icon, item):
        self.icon.stop()
        self.root.after(0, self.root.quit)

    # ── Model Loading ─────────────────────────────────────────────────────
    def _load_model(self):
        self.engine.load_model(self._set_status)

    # ── Hotkey ────────────────────────────────────────────────────────────
    def _hotkey_listener(self):
        import ctypes
        import ctypes.wintypes
        MOD = 0x0002 | 0x0001  # CTRL + ALT
        VK_F12, VK_F11 = 0x7B, 0x7A
        WM_HOTKEY = 0x0312
        ID_REC, ID_REP = 1, 2
        u32 = ctypes.windll.user32
        if not u32.RegisterHotKey(None, ID_REC, MOD, VK_F12):
            log.error("RegisterHotKey failed: Ctrl+Alt+F12")
            return
        if not u32.RegisterHotKey(None, ID_REP, MOD, VK_F11):
            log.warning("RegisterHotKey failed: Ctrl+Alt+F11")
        log.info("Hotkeys: F12=record, F11=re-paste")
        msg = ctypes.wintypes.MSG()
        while u32.GetMessageW(ctypes.byref(msg), None, 0, 0):
            if msg.message == WM_HOTKEY:
                if msg.wParam == ID_REC:
                    self._toggle()
                elif msg.wParam == ID_REP:
                    self._repaste()

    def _toggle(self):
        if not self.engine.can_toggle():
            return
        if not self.engine.is_recording:
            self._start_recording()
        else:
            self._stop_and_transcribe()

    def _repaste(self):
        if self.last_text:
            pyperclip.copy(self.last_text)
            time.sleep(0.05)
            pyautogui.hotkey("ctrl", "v")
            log.info("Re-pasted")

    # ── Recording Flow ────────────────────────────────────────────────────
    def _start_recording(self):
        dev_id = self.engine.get_device_id(self.input_devs)
        try:
            self.engine.start_recording(
                device_id=dev_id,
                on_vad_stop=lambda: self.root.after(0, self._stop_and_transcribe))
            self._set_status("recording", "Recording...")
            self.root.after(0, lambda: self._overlay.show("recording"))
            self.root.after(0, self._update_rec_bar)
        except Exception as e:
            log.error(f"Start failed: {e}")
            self._set_status("idle", "Mic error \u2014 check device")
            self.root.after(0, self._update_rec_bar)

    def _stop_and_transcribe(self):
        audio, duration = self.engine.stop_recording()
        self.root.after(0, self._update_rec_bar)
        if audio is None:
            self._set_status("idle", f"Ready [{DEVICE_LABEL}]")
            self.root.after(0, self._overlay.hide)
            return
        threading.Thread(target=self._transcribe,
                         args=(audio, duration), daemon=True).start()

    def _transcribe(self, audio, duration):
        self._set_status("transcribing", "Transcribing...")
        self.root.after(0, lambda: self._overlay.show("transcribing"))
        self.root.after(0, self._update_rec_bar)
        try:
            raw = self.engine.transcribe(
                audio,
                on_segment=lambda t: self.root.after(
                    0, lambda t=t: self._gui_update_text(t)))
            if not raw:
                return
            self._set_status("transcribing",
                             f"Cleaning [{self.settings.clean_model}]...")
            self.root.after(0, lambda: self._overlay.set_label("Cleaning..."))
            text = self.engine.clean_text(raw)
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
            log.error(f"Pipeline: {e}")
            self.last_text = f"[Error: {e}]"
            self.root.after(0, lambda: self._gui_update_text(self.last_text))
        finally:
            self._set_status("idle", f"Ready [{DEVICE_LABEL}]")
            self.root.after(0, self._overlay.hide)
            self.root.after(0, self._update_rec_bar)

    def _output_text(self, text):
        mode = self.settings.output_mode
        if mode == "Auto-paste":
            prev = ""
            try:
                prev = pyperclip.paste()
            except Exception:
                pass
            pyperclip.copy(text)
            time.sleep(0.05)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.15)
            if prev and prev != self._last_pasted:
                pyperclip.copy(prev)
            self._last_pasted = text
        elif mode == "Clipboard only":
            pyperclip.copy(text)
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
                    pyperclip.copy(text)
            else:
                pyperclip.copy(text)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    WhisperApp()
