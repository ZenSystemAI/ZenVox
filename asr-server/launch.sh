#!/bin/bash
# Launch the ZenVox ASR server on the P620, pinned to GPU 1.
cd "$HOME/asr-server" || exit 1
pkill -f "asr-server/server.py" 2>/dev/null
sleep 1
export CUDA_VISIBLE_DEVICES=1
export ZENVOX_ASR_MODEL="${ZENVOX_ASR_MODEL:-large-v3}"
export ZENVOX_ASR_PORT="${ZENVOX_ASR_PORT:-8771}"
nohup ./.venv/bin/python server.py > server.log 2>&1 &
PID=$!
echo "started pid $PID (model=$ZENVOX_ASR_MODEL port=$ZENVOX_ASR_PORT gpu=$CUDA_VISIBLE_DEVICES)"
sleep 4
echo "--- alive? ---"
if kill -0 "$PID" 2>/dev/null; then echo "process alive"; else echo "PROCESS DIED"; fi
echo "--- server.log ---"
cat server.log 2>/dev/null
