#!/usr/bin/env bash
# AWQ baseline (cyankiwi/Qwen3.8-27B-AWQ-INT4) in the stock lmsysorg/sglang:v0.5.20 image, same box, same flags family.
#   scripts/serve_awq_3090.sh <run name> [extra sglang args...]     Env: GPU (1), PORT (30001), DSPARK=0|1
set -uo pipefail
NAME=${1:?run name}; shift
GPU=${GPU:-1}; PORT=${PORT:-30001}; IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.20}
W=$HOME/rtx-3090; RUN=$W/runs/$NAME; mkdir -p "$RUN"
docker rm -f "sgl-$NAME" >/dev/null 2>&1
SPEC=""
[ "${DSPARK:-0}" = "1" ] && SPEC="--speculative-algorithm DSPARK --speculative-draft-model-path /models/Qwen3.8-27B-DSpark --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant"
INNER="exec python3 -m sglang.launch_server --model-path /hfrepo/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea --trust-remote-code --host 0.0.0.0 --port $PORT --served-model-name Qwen3.8-27B-AWQ-INT4 $SPEC"
for a in "$@"; do INNER+=" $(printf '%q' "$a")"; done
ARGV=(docker run -d --name "sgl-$NAME" --gpus "device=$GPU" --shm-size 16g --ipc=host -p 127.0.0.1:$PORT:$PORT
  -e HF_HUB_OFFLINE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -v "$HOME/models:/models:ro" -v "$HOME/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4:/hfrepo:ro" -v "$HOME/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea:/awq:ro" -v "$W:/w"
  --entrypoint bash "$IMAGE" -c "$INNER")
printf '%q ' "${ARGV[@]}" > "$RUN/serve_cmd.txt"; echo >> "$RUN/serve_cmd.txt"
"${ARGV[@]}" > "$RUN/container_id.txt" 2>&1 || { echo DOCKER_RUN_FAILED; cat "$RUN/container_id.txt"; exit 1; }
nohup docker logs -f "sgl-$NAME" > "$RUN/engine.log" 2>&1 &
echo STARTED
