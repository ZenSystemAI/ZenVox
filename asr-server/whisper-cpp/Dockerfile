# whisper.cpp STT server for the GX10 (NVIDIA GB10 / Blackwell, aarch64).
# Built from source for sm_120;121 — the validated GPU path on this hardware
# (CTranslate2/faster-whisper ships no CUDA aarch64 wheel and silently falls
# back to CPU on the GB10). See asr-server/whisper-cpp/README.md.
#
# Single-stage; nvidia/cuda is used because nvcr.io has no ARM64 CUDA tag.
# Ubuntu 24.04 matches the GX10 DGX OS glibc (2.38). Architecture "120;121"
# is required — "120" alone compiles sm_120a which is incompatible with the
# GB10. libcuda.so.1 is absent at build time, so link against the CUDA stubs.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y \
    build-essential cmake git libgomp1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN rm -rf build && \
    cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="120;121" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" && \
      cmake --build build --target whisper-server -j$(nproc)
EXPOSE 8772
ENTRYPOINT ["./build/bin/whisper-server"]
