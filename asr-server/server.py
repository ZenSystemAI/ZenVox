#!/usr/bin/env python3
"""
ZenVox remote ASR server — minimal OpenAI-compatible faster-whisper endpoint.

Runs on the P620 so dictation transcribes on a free 3090 instead of the
contended workstation GPU. Pin a card with CUDA_VISIBLE_DEVICES.

  CUDA_VISIBLE_DEVICES=1 ZENVOX_ASR_MODEL=large-v3 \
      ~/asr-server/.venv/bin/python server.py

Endpoints:
  GET  /health
  POST /v1/audio/transcriptions   (multipart: file, [language], [prompt], [hotwords])
"""
import io
import os
import sys
import ctypes
import site
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("asr")


def _preload_nvidia_libs():
    """Preload CUDA shared libs (cuBLAS/cuDNN) with RTLD_GLOBAL so ctranslate2
    finds them — LD_LIBRARY_PATH is only read at process start. Same trick
    ZenVox uses on the workstation."""
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in sorted(os.listdir(nvidia_dir)):
            lib_dir = os.path.join(nvidia_dir, pkg, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for f in sorted(os.listdir(lib_dir)):
                if f.endswith(".so") or (".so." in f and ".alt." not in f):
                    try:
                        ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


_preload_nvidia_libs()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

MODEL_NAME = os.environ.get("ZENVOX_ASR_MODEL", "large-v3")
SAMPLE_RATE = 16000

log.info(f"Loading {MODEL_NAME} on GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','all')})")
try:
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
    DEVICE = "cuda/float16"
except Exception as e:
    log.warning(f"GPU load failed ({e}); falling back to CPU int8")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    DEVICE = "cpu/int8"
log.info(f"Ready: {MODEL_NAME} on {DEVICE}")

app = FastAPI(title="ZenVox ASR")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    prompt: str = Form(None),
    hotwords: str = Form(None),
    model_name: str = Form(None, alias="model"),
):
    try:
        raw = await file.read()
        audio = decode_audio(io.BytesIO(raw), sampling_rate=SAMPLE_RATE)
        segments, info = model.transcribe(
            audio,
            language=language or None,
            beam_size=1,
            initial_prompt=prompt or None,
            hotwords=hotwords or None,
            vad_filter=True,
        )
        text = " ".join(s.text for s in segments).strip()
        log.info(f"Transcribed {len(audio)/SAMPLE_RATE:.1f}s -> {text[:80]!r}")
        return {"text": text}
    except Exception as e:
        log.exception("Transcription failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ZENVOX_ASR_PORT", "8771"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
