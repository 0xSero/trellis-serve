#!/usr/bin/env bash
# Serve an EXL3 checkpoint with SGLang inside the dev image on GPU 0.
#   scripts/serve_3090.sh <model dir under ~/models> <run name> [extra sglang args...]
# Env: GPU (default 0), PORT (30000), IMAGE (sglang-exl3:dev), KERNEL (auto|exllamav3|marlin); every SGLANG_* variable
# (plugin knobs SGLANG_EXL3_*, engine knobs such as SGLANG_ENABLE_OVERLAP_PLAN_STREAM) is forwarded into the container.
set -uo pipefail
MODEL=${1:?model dir name}; NAME=${2:?run name}; shift 2
GPU=${GPU:-0}; PORT=${PORT:-30000}; IMAGE=${IMAGE:-sglang-exl3:dev}
W=${W:-$HOME/rtx-3090}; RUN=$W/runs/$NAME; mkdir -p "$RUN"
docker rm -f "${CPFX:-sgl-}$NAME" >/dev/null 2>&1
INNER="pip install -q --no-deps -e /opt/sglang-exl3 >/dev/null 2>&1; exec python3 -m sglang.launch_server --model-path /models/$MODEL --quantization exl3 --trust-remote-code --host 0.0.0.0 --port $PORT --served-model-name $MODEL"
for a in "$@"; do INNER+=" $(printf '%q' "$a")"; done
ARGV=(docker run -d --name "${CPFX:-sgl-}$NAME" --gpus "device=$GPU" --shm-size 16g --ipc=host
  -p 127.0.0.1:$PORT:$PORT
  -e HF_HUB_OFFLINE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e SGLANG_EXL3_KERNEL=${KERNEL:-auto} $(env | grep -E "^SGLANG_|^PYTORCH_CUDA_ALLOC_CONF=" | grep -v "^SGLANG_EXL3_KERNEL=" | sed "s/^/-e /" | tr "\n" " ")
  -e PYTHONPATH=/opt/sglang-exl3/csrc/build/lib -e SGLANG_EXL3_MODEL_PATH=/models/$MODEL
  -v "$HOME/models:/models:ro" -v "$W/sglang-exl3:/opt/sglang-exl3" -v "$W:/w"
  --entrypoint bash "$IMAGE" -c "$INNER")
printf '%q ' "${ARGV[@]}" > "$RUN/serve_cmd.txt"; echo >> "$RUN/serve_cmd.txt"
"${ARGV[@]}" > "$RUN/container_id.txt" 2>&1 || { echo DOCKER_RUN_FAILED; cat "$RUN/container_id.txt"; exit 1; }
echo "container $(cat $RUN/container_id.txt) -> $RUN"
nohup docker logs -f "${CPFX:-sgl-}$NAME" > "$RUN/engine.log" 2>&1 &
echo STARTED
