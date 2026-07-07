# whisper.cpp STT server (GX10 / NVIDIA GB10)

Fully-local GPU transcription for ZenVox on the ASUS Ascent GX10
(NVIDIA GB10 / Blackwell, aarch64, CUDA 13). Audio never leaves the LAN.

This is the **GX10 path**. The sibling `asr-server/server.py`
(faster-whisper) is the **x86-only** option (e.g. the P620 4×3090 box) and is
not used here — CTranslate2 ships no CUDA aarch64 wheel and silently falls
back to CPU on the GB10.

## Why whisper.cpp
Built from source for `sm_120;121`, the validated GPU recipe on this exact
hardware (NVIDIA dev-forum, May 2026). `CMAKE_CUDA_ARCHITECTURES="120;121"`
is required: `120` alone compiles `sm_120a`, which is incompatible with the
GB10 (compute capability 12.1).

## Why Whisper not Parakeet
ZenVox's core use case is bilingual franglais (mid-sentence FR↔EN
code-switching). Parakeet TDT v3 is faster but one-language-per-utterance,
European- (not Quebec-) French, and has no hotword/initial-prompt biasing.

## Build + run (on the GX10)
```bash
ssh gx10
git clone --depth 1 --branch v1.8.4 https://github.com/ggml-org/whisper.cpp.git ~/zenvox-asr-build
cd ~/zenvox-asr-build
# drop in the three files from asr-server/whisper-cpp/
#   (Dockerfile, docker-compose.yml, .dockerignore)
mkdir -p models && wget -P ./models \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
docker compose up -d --build
curl -fsS http://localhost:8772/health        # expect compute capability 12.1, CUDA0 backend
```

## Confirm the GPU was used
The container log must show:
```
Device 0: NVIDIA GB10, compute capability 12.1
whisper_backend_init_gpu: using CUDA0 backend
```
If it shows CPU / `no kernel image available`, the `sm_120;121` build failed —
confirm `nvidia-smi` reports compute capability 12.1 and that
`CMAKE_CUDA_ARCHITECTURES` was `"120;121"` (not `"120"`).

## Transcribe (OpenAI-compatible)
The `--inference-path /v1/audio/transcriptions` flag exposes the OpenAI-style
endpoint ZenVox already calls. ZenVox's client sends `response_format=json`
so whisper-server returns `{"text": ...}`:
```bash
curl -fsS http://localhost:8772/v1/audio/transcriptions \
  -F file=@/path/to/clip.wav -F response_format=json
# → {"text":" And so my fellow Americans..."}
```
Then point ZenVox at it: **Settings → Transcription → Remote URL =
`http://<GX10_ADDR>:8772/v1`**.

> **Model bind-mount:** the model is **not** baked into the image. The
> `docker-compose.yml` mounts `./models:/app/models:ro`, so the
> `ggml-large-v3-turbo.bin` downloaded in the runbook is served from the host —
> no 1.6GB image layer, and the model can be swapped without rebuilding.

## Cleaning endpoint (deploy time)
ZenVox's **Local** cleaning provider needs a separate OpenAI-compatible LLM on the
LAN. On this GX10, a dedicated **llama.cpp** server serves `Qwen3.5-4B-Instruct`
(Q4_K_M GGUF, CUDA `sm_120;121`) on `:8088` — built the same way as whisper.cpp
(same CUDA arch flags). It runs as a systemd service (`llama-server.service`),
bound `0.0.0.0`, reachable over Tailscale.

Qwen3.5-4B-Instruct was chosen over the box's existing vLLM `nex-n2-mini` (`:8003`)
because the latter translated franglais to English instead of preserving the
language mix — Qwen3.5 follows the "do NOT translate" instruction correctly.
`--reasoning off` is required (Qwen3.5 defaults to thinking mode, which adds
`<think>` blocks and latency for a simple cleaning task).

In ZenVox: **Settings → AI Cleaning → Provider = Local**, endpoint =
`http://<GX10_ADDR>:8088/v1`, model = `qwen3.5-4b-instruct`.

(`:8003` vLLM / `nex-n2-mini` and `:8001` loopback redaction gate also exist on
this box but are not used for ZenVox cleaning.)

## Contingencies
- If `whisper-server` rejects the multi-segment `--inference-path`, fall back
  to the default `/inference` path: launch with `--inference-path /inference`,
  set ZenVox's Remote URL to `http://<GX10_ADDR>:8772` (no `/v1` suffix), and
  change `_transcribe_remote`'s URL build from `/audio/transcriptions` to
  `/inference`.
- As a last resort for a broken `sm_120;121` build, the prebuilt
  `mekopa/whisperx-blackwell` image works on the GB10 but exposes a different
  `/transcribe` API — wiring that requires changing the Phase-A5 client to
  target that API instead.
