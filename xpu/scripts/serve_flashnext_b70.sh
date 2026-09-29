#!/usr/bin/env bash
# Qwen3.8-Flash-Next EXL3 (3.05bpw_h5_ng5) on ONE Arc Pro B70 through SGLang v0.5.20-xpu + exl3xpu:
#   routed experts in USM host memory (tier 1) + device-managed LRU expert cache (EXL3_MOE_SLOTS slots, EXL3_MOE_CACHE=1),
#   n-gram table in USM host memory (zero-copy gather + decode; EXL3_NGRAM_TIER=nvme: tier 2 = NVMe + 8 GB RAM row
#   cache, needs exl3xpu/exl3xpu_ngram_nvme.so from scripts/build_ngram_nvme.sh), dense EXL3 linears via exl3xpu_C.
# Foreground (run it in tmux):  serve_flashnext_b70.sh <run dir> [extra sglang args...]
# Env: CARD (ZE_AFFINITY_MASK, default 1), PORT (30250), NAME (ftx-srv), IMAGE, MODEL, CTX, MEMFRAC, CHUNK, KVDTYPE,
#      SLOTS (EXL3_MOE_SLOTS), MAXREQ; EXL3_* / SGLANG_* are forwarded.
set -uo pipefail
RUN=${1:?run dir}; shift; mkdir -p "$RUN"
CARD=${CARD:-1}; PORT=${PORT:-30250}; NAME=${NAME:-ftx-srv}; IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.20-xpu}
MODEL=${MODEL:-turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
TS=${TS:-$HOME/freetoken-exl3/kernels/xpu_bmg/trellis-serve}
X=/opt/trellis-serve/xpu/exl3xpu
INNER="pip install -q --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/xpu >/tmp/pip.log 2>&1 || cat /tmp/pip.log; \
pip install -q --no-deps onednn==2026.0.0 >/dev/null 2>&1; \
python3 -c 'import torch; print(\"card uuid\", torch.xpu.get_device_properties(0).uuid)'; \
exec python3 -m sglang.launch_server --model-path /models/$MODEL --quantization exl3 --trust-remote-code --device xpu \
 --host 0.0.0.0 --port $PORT --served-model-name flashnext --disable-shared-experts-fusion \
 --kv-cache-dtype ${KVDTYPE:-fp8_e4m3} --context-length ${CTX:-262144} --mem-fraction-static ${MEMFRAC:-0.80} \
 --chunked-prefill-size ${CHUNK:-4096} --max-running-requests ${MAXREQ:-2} --dtype bfloat16"
for a in "$@"; do INNER+=" $(printf '%q' "$a")"; done
docker rm -f "$NAME" >/dev/null 2>&1
ARGV=(docker run --rm --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=$CARD
  --ipc=host --shm-size 64g --network host
  -e HF_HUB_OFFLINE=1 -e EXL3_MODEL_PATH=/models/$MODEL -e EXL3_LIB=$X/_C_sgl.so -e EXL3_MOE_LIB=$X/_moe_sgl.so
  -e EXL3_MOE_SLOTS=${SLOTS:-2048} -e EXL3_MOE_CACHE=${EXL3_MOE_CACHE:-1} -e TORCHINDUCTOR_COMPILE_THREADS=1
  $(env | grep -E "^(SGLANG_|EXL3_|PYTORCH_)" | grep -v -E "^(EXL3_MOE_CACHE|EXL3_LIB|EXL3_MOE_LIB)=" | sed "s/^/-e /" | tr "\n" " ")
  -v "$HOME/models:/models:ro" -v "$TS:/opt/trellis-serve" -v "$HOME/freetoken-exl3:/w"
  --entrypoint bash "$IMAGE" -c "$INNER")
printf '%q ' "${ARGV[@]}" > "$RUN/cmd.txt"; echo >> "$RUN/cmd.txt"
echo "serving on 127.0.0.1:$PORT (card $CARD), log $RUN/log.txt"
"${ARGV[@]}" > "$RUN/log.txt" 2>&1
echo "exit $?" >> "$RUN/log.txt"
