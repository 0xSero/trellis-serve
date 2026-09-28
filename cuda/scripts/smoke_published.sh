#!/usr/bin/env bash
# Boot the PUBLISHED image exactly as the registry recipe launches it (entrypoint shim + argv, per-model mounts).
#   scripts/smoke_published.sh <digest> <model dir> <run name> [sglang args...]
# Env: GPU (0), PORT (30000), DRAFT (optional draft checkpoint dir under ~/models, mounted at /models/$DRAFT), and every
# SGLANG_* variable is forwarded (e.g. SGLANG_EXL3_EMBED_HOST=1).
set -uo pipefail
DIGEST=${1:?digest}; MODEL=${2:?model dir}; NAME=${3:?run name}; shift 3
GPU=${GPU:-0}; PORT=${PORT:-30000}; W=$HOME/rtx-3090; RUN=$W/runs/$NAME; mkdir -p "$RUN"
timeout 30s docker rm -f "${CPFX:-sgl-}$NAME" >/dev/null 2>&1
ARGV=(docker run -d --name "${CPFX:-sgl-}$NAME" --gpus "device=$GPU" --shm-size 16g -p 127.0.0.1:$PORT:$PORT
  -e HF_HUB_OFFLINE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID $(env | grep -E "^SGLANG_|^PYTORCH_CUDA_ALLOC_CONF=" | sed "s/^/-e /" | tr "\n" " ")
  -v "$HOME/models/$MODEL:/models/$MODEL:ro" ${DRAFT:+-v "$HOME/models/$DRAFT:/models/$DRAFT:ro"}
  "ghcr.io/0xsero/sglang-exl3@$DIGEST" python3 -m sglang.launch_server --model-path /models/$MODEL --quantization exl3
  --trust-remote-code --host 0.0.0.0 --port $PORT --served-model-name "$MODEL" "$@")
printf '%q ' "${ARGV[@]}" > "$RUN/serve_cmd.txt"; echo >> "$RUN/serve_cmd.txt"
timeout 60s "${ARGV[@]}" > "$RUN/container_id.txt" 2>&1 || { echo DOCKER_RUN_FAILED; cat "$RUN/container_id.txt"; exit 1; }
nohup docker logs -f "${CPFX:-sgl-}$NAME" > "$RUN/engine.log" 2>&1 &
echo $! > "$RUN/engine.log.pid"
echo STARTED
