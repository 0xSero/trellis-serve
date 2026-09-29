#!/usr/bin/env bash
# Qwen3.8-Flash-Next EXL3 (3.05bpw_h5_ng5) on ONE RTX 3090 through SGLang + trellis-serve.
# Default (best, 2026-09-29): routed experts in pinned host memory + a global GPU expert cache (kernel lane K05,
#   SGLANG_EXL3_MOE_OFFLOAD=gpu_cache, 10.5 GB = 5,637 slots, prefill staging in 4 parts), token embedding in pinned host
#   memory, EXL3 n-gram table on NVMe (pread x32) behind an 8 GB RAM row cache (SGLANG_EXL3_NGRAM_TIER=nvme; =pinned keeps all 32.6 GB in pinned RAM, +4 % prefill),
#   fp8 e4m3 KV for 210k tokens (context 204,800), 8k prefill chunks, decode CUDA graphs.
# Fallback: SGLANG_EXL3_MOE_OFFLOAD=cpu (exllamav3 CPU worker, graph-safe handoff) with CHUNK=16384 CTX=262144
#   MAXTOK=270336 MEMFRAC=0.80 SGLANG_EXL3_EMBED_HOST=0.
# Foreground (run it under bench/gpu1_run.sh in tmux so the GPU lock is held exactly as long as the server lives):
#   serve_flashnext_3090.sh <run dir> [extra sglang args...]
# Env: GPU (1), PORT (30100), IMAGE (sglang-exl3:dev), NAME (ft-int-serve), MODEL (dir under ~/models),
#      CTX (204800), MEMFRAC (0.88), CHUNK (8192), THREADS (24), GRAPH_BS (8), KVDTYPE (fp8_e4m3), MAMBA (16), MAXTOK (210000 KV tokens), CACHE_GB (auto = free VRAM - staging - 8.9 GB reserve; 10.5 GB on omarchy),
#      DOCKER_ENV (extra "-e K=V ..." docker args); SGLANG_* / EXL3_* are forwarded.
set -uo pipefail
RUN=${1:?run dir}; shift; mkdir -p "$RUN"
GPU=${GPU:-1}; PORT=${PORT:-30100}; IMAGE=${IMAGE:-sglang-exl3:dev}; NAME=${NAME:-ft-int-serve}
MODEL=${MODEL:-turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
TS=${TS:-$HOME/freetoken-exl3/integration/trellis-serve}
INNER="pip install -q --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/cuda >/tmp/pip.log 2>&1 || cat /tmp/pip.log; \
exec python3 -m sglang.launch_server --model-path /models/$MODEL --quantization exl3 --trust-remote-code \
 --host 0.0.0.0 --port $PORT --served-model-name flashnext --disable-shared-experts-fusion \
 --kv-cache-dtype ${KVDTYPE:-fp8_e4m3} --context-length ${CTX:-204800} --mem-fraction-static ${MEMFRAC:-0.88} \
 --chunked-prefill-size ${CHUNK:-8192} --max-running-requests ${MAXREQ:-4} --cuda-graph-max-bs-decode ${GRAPH_BS:-8} \
 --cuda-graph-backend-prefill disabled --max-mamba-cache-size ${MAMBA:-16} --max-total-tokens ${MAXTOK:-210000}"
for a in "$@"; do INNER+=" $(printf '%q' "$a")"; done
docker rm -f "$NAME" >/dev/null 2>&1
ARGV=(docker run --rm --name "$NAME" --gpus "device=$GPU" --ipc=host --shm-size 64g --ulimit memlock=-1
  -p 127.0.0.1:$PORT:$PORT
  -e HF_HUB_OFFLINE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e SGLANG_EXL3_MODEL_PATH=/models/$MODEL
  -e SGLANG_EXL3_MOE_OFFLOAD=${SGLANG_EXL3_MOE_OFFLOAD:-gpu_cache} -e EXL3_MOE_CPU_THREADS=${THREADS:-24}
  -e SGLANG_EXL3_EXPERT_CACHE_GB=${CACHE_GB:-auto} -e SGLANG_EXL3_EMBED_HOST=${SGLANG_EXL3_EMBED_HOST:-1}
  -e SGLANG_EXL3_OFFLOAD_STAGING_PARTS=${SGLANG_EXL3_OFFLOAD_STAGING_PARTS:-4} -e SGLANG_EXL3_OFFLOAD_FUSED=${SGLANG_EXL3_OFFLOAD_FUSED:-1}
  -e SGLANG_EXL3_NGRAM_TIER=${SGLANG_EXL3_NGRAM_TIER:-nvme}
  -e PYTHONPATH=/opt/trellis-serve/cuda/csrc/build/lib -e SGLANG_EXL3_JIT_DIR=/opt/trellis-serve/cuda/csrc/build/jit
  $(env | grep -E "^(SGLANG_|EXL3_|PYTORCH_CUDA_ALLOC_CONF=)" | grep -v -E "^(SGLANG_EXL3_MOE_OFFLOAD|EXL3_MOE_CPU_THREADS|SGLANG_EXL3_EXPERT_CACHE_GB|SGLANG_EXL3_EMBED_HOST|SGLANG_EXL3_NGRAM_TIER|SGLANG_EXL3_OFFLOAD_STAGING_PARTS|SGLANG_EXL3_OFFLOAD_FUSED)=" | sed "s/^/-e /" | tr "\n" " ")
  ${DOCKER_ENV:-}
  -v "$HOME/models:/models:ro" -v "$TS:/opt/trellis-serve" -v "$HOME/freetoken-exl3:/w"
  --entrypoint bash "$IMAGE" -c "$INNER")
printf '%q ' "${ARGV[@]}" > "$RUN/cmd.txt"; echo >> "$RUN/cmd.txt"
echo "serving on 127.0.0.1:$PORT, log $RUN/log.txt"
"${ARGV[@]}" > "$RUN/log.txt" 2>&1
echo "exit $?" >> "$RUN/log.txt"
