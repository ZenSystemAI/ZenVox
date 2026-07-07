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

import asyncio

from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

MODEL_NAME = os.environ.get("ZENVOX_ASR_MODEL", "large-v3")
SAMPLE_RATE = 16000
# Optional shared secret: set ZENVOX_ASR_TOKEN on the server and the same value
# in the client to keep other LAN devices from burning GPU time here.
AUTH_TOKEN = os.environ.get("ZENVOX_ASR_TOKEN", "")
# 20 min of 16-bit 16 kHz WAV is ~37 MB; cap uploads well above that but far
# below anything that could exhaust RAM on a shared box.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

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
# The model is not safe for concurrent transcribe calls; serialize explicitly
# and run the blocking decode in a worker thread so /health and other requests
# aren't frozen behind a long clip (the old async-def handler blocked the
# whole event loop for the duration of every transcription).
_asr_lock = asyncio.Lock()


def _do_transcribe(audio, language, prompt, hotwords):
    segments, info = model.transcribe(
        audio,
        language=language or None,
        beam_size=1,
        initial_prompt=prompt or None,
        hotwords=hotwords or None,
        vad_filter=True,
    )
    return " ".join(s.text for s in segments).strip()


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
    x_zenvox_key: str = Header(None),
):
    if AUTH_TOKEN and x_zenvox_key != AUTH_TOKEN:
        return JSONResponse(status_code=401, content={"error": "bad or missing X-ZenVox-Key"})
    try:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"error": "upload too large"})
        audio = decode_audio(io.BytesIO(raw), sampling_rate=SAMPLE_RATE)
        async with _asr_lock:
            text = await run_in_threadpool(_do_transcribe, audio, language, prompt, hotwords)
        log.info(f"Transcribed {len(audio)/SAMPLE_RATE:.1f}s ({len(text)} chars)")
        log.debug(f"Text: {text[:80]!r}")
        return {"text": text}
    except Exception as e:
        log.exception("Transcription failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ZENVOX_ASR_PORT", "8771"))
    # Default to loopback — binding every interface exposed an unauthenticated
    # GPU endpoint to the whole LAN. Set ZENVOX_ASR_HOST to the tailnet IP (or
    # 0.0.0.0 behind a firewall) to serve other machines.
    host = os.environ.get("ZENVOX_ASR_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="warning")
