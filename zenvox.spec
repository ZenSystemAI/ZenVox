# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for ZenVox — Voice to text, cleaned by AI.
Builds a one-directory distribution with CUDA + Whisper support.

Build: pyinstaller zenvox.spec
Output: dist/ZenVox/
"""
import os
import site
import glob

block_cipher = None

# ── Paths ────────────────────────────────────────────────────────────────────
sp = site.getsitepackages()[0]

# ── Collect NVIDIA CUDA DLLs ────────────────────────────────────────────────
nvidia_bins = []
nvidia_dir = os.path.join(sp, 'nvidia')
if os.path.isdir(nvidia_dir):
    for pkg in os.listdir(nvidia_dir):
        bin_dir = os.path.join(nvidia_dir, pkg, 'bin')
        if os.path.isdir(bin_dir):
            for dll in os.listdir(bin_dir):
                if dll.endswith('.dll'):
                    nvidia_bins.append((os.path.join(bin_dir, dll), '.'))

# ── Collect ctranslate2 DLLs ────────────────────────────────────────────────
ct2_dir = os.path.join(sp, 'ctranslate2')
ct2_bins = []
if os.path.isdir(ct2_dir):
    for f in os.listdir(ct2_dir):
        if f.endswith('.dll'):
            ct2_bins.append((os.path.join(ct2_dir, f), '.'))

# ── Collect Silero VAD model ────────────────────────────────────────────────
import faster_whisper
fw_pkg_dir = os.path.dirname(faster_whisper.__file__)
vad_onnx = os.path.join(fw_pkg_dir, 'assets', 'silero_vad_v6.onnx')
vad_data = []
if os.path.exists(vad_onnx):
    vad_data.append((vad_onnx, os.path.join('faster_whisper', 'assets')))
else:
    print(f"WARNING: VAD model not found at {vad_onnx}")

# ── Collect customtkinter theme files ───────────────────────────────────────
ctk_dir = None
for _sp in site.getsitepackages():
    _ctk = os.path.join(_sp, 'Lib', 'site-packages', 'customtkinter')
    if os.path.isdir(_ctk):
        ctk_dir = _ctk
        break
    _ctk = os.path.join(_sp, 'customtkinter')
    if os.path.isdir(_ctk):
        ctk_dir = _ctk
        break
if ctk_dir is None:
    import customtkinter
    ctk_dir = os.path.dirname(customtkinter.__file__)

# ── Collect onnxruntime DLLs ────────────────────────────────────────────────
ort_dir = os.path.join(sp, 'onnxruntime')
ort_bins = []
if os.path.isdir(ort_dir):
    for root, dirs, files in os.walk(ort_dir):
        for f in files:
            if f.endswith(('.dll', '.so')):
                rel = os.path.relpath(root, sp)
                ort_bins.append((os.path.join(root, f), rel))

a = Analysis(
    ['whisper.py'],
    pathex=[],
    binaries=nvidia_bins + ct2_bins + ort_bins,
    datas=[
        ('whisper.ico', '.'),
        ('icon_concepts/zenvox_icon_final.png', 'icon_concepts'),
    ] + vad_data + [
        (ctk_dir, 'customtkinter'),
    ],
    hiddenimports=[
        'customtkinter',
        'pystray._win32',
        'PIL._tkinter_finder',
        'onnxruntime',
        'ctranslate2',
        'faster_whisper',
        'huggingface_hub',
        'sounddevice',
        'pyperclip',
        'pyautogui',
        'google.generativeai',
        'openai',
        'anthropic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'matplotlib', 'scipy', 'pandas',
        'pytest', 'setuptools', 'pip',
        'tkinter.test', 'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ZenVox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # No console window — windowed app
    disable_windowed_traceback=False,
    icon='whisper.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='ZenVox',
)
