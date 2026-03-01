#!/usr/bin/env python3
"""
whisper.py  —  Voice to text, system tray + GUI
Press Ctrl+Alt+F12 (G1) to start/stop recording. Transcribes and pastes.
Right-click tray icon (or open window) for settings and last result.
"""
import json
import threading
import time
import customtkinter as ctk
from pathlib import Path

import numpy as np
import pyautogui
import pyperclip
import keyboard
import pystray
import sounddevice as sd
from PIL import Image, ImageDraw

# ── Config ──────────────────────────────────────────────────────────────────
HOTKEY      = "ctrl+alt+f12"
SAMPLE_RATE = 16000
MODELS      = ["tiny", "base", "small", "large-v3-turbo"]
LANGS       = {
    "Auto-detect":   None,
    "English":       "en",
    "Français":      "fr",
    "Français (CA)": "fr",   # same code, gets initial_prompt for QC bias
}

# ── Settings persistence ─────────────────────────────────────────────────────
SETTINGS_FILE = Path(__file__).parent / "settings.json"

def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_settings(data: dict):
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

# ── GPU detection ─────────────────────────────────────────────────────────────
DEVICE  = "cpu"
COMPUTE = "int8"
try:
    import os, site, ctypes, ctranslate2
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if not os.path.isdir(bin_dir):
                continue
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            for dll in os.listdir(bin_dir):
                if dll.endswith(".dll"):
                    try:
                        ctypes.CDLL(os.path.join(bin_dir, dll))
                    except Exception:
                        pass
    if ctranslate2.get_cuda_device_count() > 0:
        DEVICE  = "cuda"
        COMPUTE = "float16"   # optimal for RTX 4080 Super
except Exception:
    pass

# ── Hardware tuning ───────────────────────────────────────────────────────────
# RTX 4080 Super: large batches, float16 already set above
# AMD 9950X: 16 P-cores + 16 E-cores = plenty of threads for CPU fallback
CPU_THREADS  = 16   # for CPU inference / audio preprocessing on 9950X
NUM_WORKERS  = 4    # parallel segment decoding workers

# ── Tray icons (colored circles) ─────────────────────────────────────────────
def _circle(color):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([4, 4, 60, 60], fill=color)
    return img

ICONS = {
    "idle":         _circle("#4caf50"),   # green
    "recording":    _circle("#ef5350"),   # red
    "transcribing": _circle("#ff9800"),   # orange
    "loading":      _circle("#9e9e9e"),   # gray
}

def _list_input_devices():
    devs, seen = [], set()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["name"] not in seen:
            devs.append((i, d["name"]))
            seen.add(d["name"])
    return devs


class WhisperTray:
    def __init__(self):
        self.model         = None
        self.is_recording  = False
        self.audio_chunks  = []
        self.stream        = None
        self.input_devs    = _list_input_devices()
        self.last_text     = ""
        self._transcribing = False
        self._last_toggle  = 0.0
        self._status_text  = "Loading..."

        # Load persisted settings
        saved = _load_settings()
        self.model_name = saved.get("model_name", "large-v3-turbo")  # 4080 Super can handle it
        self.lang_name  = saved.get("lang_name",  "Auto-detect")
        self.mic_name   = saved.get("mic_name",   self.input_devs[0][1] if self.input_devs else "default")

        # ── Build CustomTkinter window (hidden on start) ────────────────────
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("Whisper Desktop Tool")
        self.root.geometry("640x500")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)  # close → hide to tray
        ico_path = Path(__file__).parent / "whisper.ico"
        if ico_path.exists():
            self.root.iconbitmap(str(ico_path))
        self._build_gui()
        self.root.withdraw()  # start hidden

        # ── Build tray ──────────────────────────────────────────────────────
        self.icon = pystray.Icon(
            "whisper",
            ICONS["loading"],
            "Whisper — Loading...",
            menu=self._build_menu()
        )

        threading.Thread(target=self._load_model,      daemon=True).start()
        threading.Thread(target=self._hotkey_listener, daemon=True).start()

        self.icon.run_detached()  # tray in background thread; tkinter owns main thread
        self.root.mainloop()

    # ── GUI ────────────────────────────────────────────────────────────────
    def _build_gui(self):
        bg_hero  = "#141414"
        bg_panel = "#1C2523"
        fg_text  = "#FFFFFF"
        fg_muted = "#9CA3AF"
        teal     = "#2B7269"
        teal_h   = "#3A9188"
        border   = "#2A332F"
        
        self.root.configure(fg_color=bg_hero)

        # Global container
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(fill="both", expand=True)

        # 1. Floating top navbar
        self.gui_status_var = ctk.StringVar(value="Whisper — Loading...")
        status_frame = ctk.CTkFrame(main_frame, fg_color=bg_panel, border_color=border, border_width=1, corner_radius=12)
        status_frame.pack(fill="x", padx=24, pady=(24, 12))
        
        status_inner = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_inner.pack(fill="x", padx=16, pady=12)
        
        ctk.CTkLabel(status_inner, text="◆ Whisper", font=("Segoe UI", 16, "bold"), text_color="#FFFFFF").pack(side="left")
        ctk.CTkLabel(status_inner, textvariable=self.gui_status_var, font=("Segoe UI", 13), text_color=teal).pack(side="right")

        # 2. Hero Action Pane
        text_bg_frame = ctk.CTkFrame(main_frame, fg_color=bg_panel, border_color=border, border_width=1, corner_radius=16)
        text_bg_frame.pack(fill="both", expand=True, padx=24, pady=(8, 12))

        header_frame = ctk.CTkFrame(text_bg_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(16, 4))
        ctk.CTkLabel(header_frame, text="Last transcription", font=("Segoe UI", 13, "bold"), text_color=fg_muted).pack(side="left")
        
        self.gui_text = ctk.CTkTextbox(text_bg_frame, wrap="word", state="disabled",
                                       fg_color="transparent", text_color=fg_text, font=("Segoe UI", 14))
        self.gui_text.pack(fill="both", expand=True, padx=12, pady=4)

        # 3. Floating Copy Button
        btn_frame = ctk.CTkFrame(text_bg_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(8, 16))
        
        copy_btn = ctk.CTkButton(btn_frame, text="Copy", command=self._gui_copy,
                                 fg_color=teal, hover_color=teal_h, text_color="#FFFFFF",
                                 font=("Segoe UI", 13, "bold"), corner_radius=8, width=110, height=36)
        copy_btn.pack(side="right")

        # 4. Settings Grid
        settings_frame = ctk.CTkFrame(main_frame, fg_color=bg_panel, corner_radius=12, border_color=border, border_width=1)
        settings_frame.pack(fill="x", padx=24, pady=(8, 12))
        
        grid_inner = ctk.CTkFrame(settings_frame, fg_color="transparent")
        grid_inner.pack(fill="x", padx=20, pady=16)
        
        # Model
        self.gui_model_var = ctk.StringVar(value=self.model_name)
        cb_model = ctk.CTkComboBox(grid_inner, variable=self.gui_model_var, values=MODELS,
                                   fg_color=bg_hero, border_color=border, button_color=border,
                                   button_hover_color=teal, dropdown_fg_color=bg_panel, dropdown_hover_color=teal,
                                   font=("Segoe UI", 12), text_color=fg_text, width=160, command=self._gui_model_changed)
        cb_model.pack(side="left", padx=(0, 16))
        
        # Language
        self.gui_lang_var = ctk.StringVar(value=self.lang_name)
        cb_lang = ctk.CTkComboBox(grid_inner, variable=self.gui_lang_var, values=list(LANGS.keys()),
                                  fg_color=bg_hero, border_color=border, button_color=border,
                                  button_hover_color=teal, dropdown_fg_color=bg_panel, dropdown_hover_color=teal,
                                  font=("Segoe UI", 12), text_color=fg_text, width=150, command=self._gui_lang_changed)
        cb_lang.pack(side="left", padx=(0, 16))
        
        # Mic
        self.gui_mic_var = ctk.StringVar(value=self.mic_name)
        mic_names = [n for _, n in self.input_devs]
        cb_mic = ctk.CTkComboBox(grid_inner, variable=self.gui_mic_var, values=mic_names,
                                 fg_color=bg_hero, border_color=border, button_color=border,
                                 button_hover_color=teal, dropdown_fg_color=bg_panel, dropdown_hover_color=teal,
                                 font=("Segoe UI", 12), text_color=fg_text, command=self._gui_mic_changed)
        cb_mic.pack(side="left", fill="x", expand=True)

        # 5. Footer Hotkey
        footer_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        footer_frame.pack(fill="x", padx=32, pady=(0, 16))
        ctk.CTkLabel(footer_frame, text=f"Hotkey: {HOTKEY}", font=("Segoe UI", 11), text_color=fg_muted).pack(side="left")

    def _gui_copy(self):
        if self.last_text:
            pyperclip.copy(self.last_text)

    def _gui_model_changed(self, value=None):
        self.model_name = self.gui_model_var.get()
        self._persist_settings()
        self._reload_model()

    def _gui_lang_changed(self, value=None):
        self.lang_name = self.gui_lang_var.get()
        self._persist_settings()

    def _gui_mic_changed(self, value=None):
        self.mic_name = self.gui_mic_var.get()
        self._persist_settings()

    def _gui_update_text(self, text):
        """Thread-safe update of the transcription text box."""
        self.gui_text.configure(state="normal")
        self.gui_text.delete("1.0", "end")
        self.gui_text.insert("end", text)
        self.gui_text.configure(state="disabled")

    # ── Settings ───────────────────────────────────────────────────────────
    def _persist_settings(self):
        _save_settings({
            "model_name": self.model_name,
            "lang_name":  self.lang_name,
            "mic_name":   self.mic_name,
        })

    # ── Tray menu ──────────────────────────────────────────────────────────
    def _build_menu(self):
        def model_action(name):
            def action(icon, item):
                self.model_name = name
                self._persist_settings()
                self.root.after(0, lambda: self.gui_model_var.set(name))
                self._reload_model()
            return action

        def lang_action(name):
            def action(icon, item):
                self.lang_name = name
                self._persist_settings()
                self.root.after(0, lambda: self.gui_lang_var.set(name))
            return action

        def mic_action(name):
            def action(icon, item):
                self.mic_name = name
                self._persist_settings()
                self.root.after(0, lambda: self.gui_mic_var.set(name))
            return action

        model_items = [
            pystray.MenuItem(m, model_action(m),
                             checked=lambda item, m=m: self.model_name == m,
                             radio=True)
            for m in MODELS
        ]
        lang_items = [
            pystray.MenuItem(l, lang_action(l),
                             checked=lambda item, l=l: self.lang_name == l,
                             radio=True)
            for l in LANGS
        ]
        mic_items = [
            pystray.MenuItem(n, mic_action(n),
                             checked=lambda item, n=n: self.mic_name == n,
                             radio=True)
            for _, n in self.input_devs
        ]

        return pystray.Menu(
            pystray.MenuItem(
                lambda item: f"Last: {self.last_text[:50]}..." if len(self.last_text) > 50
                             else f"Last: {self.last_text}" if self.last_text
                             else "Last: -",
                self._copy_last
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Window", lambda icon, item: self.root.after(0, self.root.deiconify)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Model",    pystray.Menu(*model_items)),
            pystray.MenuItem("Language", pystray.Menu(*lang_items)),
            pystray.MenuItem("Mic",      pystray.Menu(*mic_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    def _copy_last(self, icon, item):
        if self.last_text:
            pyperclip.copy(self.last_text)

    def _quit(self, icon, item):
        self.icon.stop()
        self.root.after(0, self.root.quit)

    # ── Status ─────────────────────────────────────────────────────────────
    def _set_status(self, state, tooltip=None):
        msg = tooltip or f"Whisper — {state}"
        self._status_text = msg          # stored for tray menu display
        self.icon.icon  = ICONS.get(state, ICONS["idle"])
        self.icon.title = msg
        self.root.after(0, lambda: self.gui_status_var.set(msg))

    # ── Model loading ──────────────────────────────────────────────────────
    def _load_model(self):
        from faster_whisper import WhisperModel
        self._set_status("loading", f"Whisper — Loading {self.model_name}...")
        self.model = WhisperModel(
            self.model_name,
            device=DEVICE,
            compute_type=COMPUTE,
            cpu_threads=CPU_THREADS,   # 9950X: use plenty of cores
            num_workers=NUM_WORKERS,
        )
        dev = "GPU (4080S)" if DEVICE == "cuda" else "CPU (9950X)"
        self._set_status("idle", f"Whisper — Ready [{dev}]")

    def _reload_model(self):
        self.model = None
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── Hotkey ─────────────────────────────────────────────────────────────
    def _hotkey_listener(self):
        keyboard.add_hotkey(HOTKEY, self._toggle)
        keyboard.wait()

    def _toggle(self):
        now = time.time()
        if now - self._last_toggle < 0.4:   # debounce (G HUB fires on press+release)
            return
        self._last_toggle = now
        if self.model is None or self._transcribing:
            return
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_and_transcribe()

    # ── Recording ──────────────────────────────────────────────────────────
    def _get_device_id(self):
        for idx, name in self.input_devs:
            if name == self.mic_name:
                return idx
        return None

    def _start_recording(self):
        self.is_recording = True
        self.audio_chunks = []

        def callback(indata, frames, t, status):
            if self.is_recording:
                self.audio_chunks.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=self._get_device_id(), callback=callback
        )
        self.stream.start()
        self._set_status("recording", "Whisper — Recording...")

    def _stop_and_transcribe(self):
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        audio = (np.concatenate(self.audio_chunks, axis=0).flatten()
                 if self.audio_chunks else np.array([]))
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    # ── Transcription ──────────────────────────────────────────────────────
    def _transcribe(self, audio):
        self._transcribing = True
        self._set_status("transcribing", "Whisper — Transcribing...")
        dev = "GPU (4080S)" if DEVICE == "cuda" else "CPU (9950X)"
        try:
            if len(audio) < SAMPLE_RATE * 0.3:
                return

            lang           = LANGS.get(self.lang_name)
            initial_prompt = ("Transcription en français canadien."
                              if self.lang_name == "Français (CA)" else None)

            segments, _ = self.model.transcribe(
                audio,
                language=lang,
                beam_size=5,
                initial_prompt=initial_prompt,
                vad_filter=True,        # reduces false transcriptions from silence/noise
            )
            text = " ".join(s.text for s in segments).strip()

            if text:
                self.last_text = text
                pyperclip.copy(text)
                self.root.after(0, lambda t=text: self._gui_update_text(t))
                self.icon.menu = self._build_menu()
                time.sleep(0.05)
                pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            self.last_text = f"[Error: {e}]"
            self.root.after(0, lambda t=self.last_text: self._gui_update_text(t))
            self.icon.menu = self._build_menu()
        finally:
            self._transcribing = False
            self._set_status("idle", f"Whisper — Ready [{dev}]")


if __name__ == "__main__":
    WhisperTray()
