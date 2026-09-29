#!/bin/bash
# K08 A/B on the whole model (GPU 0): TTFT of fresh 32k prompts (8k chunks) with the overlay tree, acc32 vs acc16 MoE prefill.
set -u
cd ~/freetoken-exl3
R=${R:-runs/2026-09-29-K08-ab}; mkdir -p $R
for V in ${VARIANTS:-acc32 acc16}; do
  export SGLANG_EXL3_MOE_PREFILL_FP16_ACC=$(echo $V | grep -q acc16 && echo 1 || echo 0) SGLANG_EXL3_OFFLOAD_COMPACT_PARTS=$(echo $V | grep -q compact && echo 1 || echo 0) SGLANG_EXL3_OFFLOAD_FUSED=1
  export GPU=0 PORT=30200 NAME=ft-k08 CACHE_GB=6 MEMFRAC=0.80 TS=$HOME/freetoken-exl3/kernels/cuda_sm86/k08/ts_overlay
  integration/trellis-serve/cuda/scripts/serve_flashnext_3090.sh $R/serve_$V &
  SP=$!
  for i in $(seq 1 180); do curl -s -m 5 http://127.0.0.1:30200/health >/dev/null 2>&1 && break; grep -q "^exit" $R/serve_$V/log.txt 2>/dev/null && break; sleep 5; done
  curl -s -m 5 http://127.0.0.1:30200/health >/dev/null && echo "$V up" || { echo "$V not up"; tail -20 $R/serve_$V/log.txt; }
  python3 kernels/cuda_sm86/k08/profile_prefill.py --url http://127.0.0.1:30200 --ctx 4096 --no-profile --dir x --host-dir x > $R/ttft_$V.txt 2>&1
  for rep in 1 2 3; do python3 kernels/cuda_sm86/k08/profile_prefill.py --url http://127.0.0.1:30200 --ctx 32768 --no-profile --dir x --host-dir x; done >> $R/ttft_$V.txt 2>&1
  # quality probe: greedy continuation of a fixed natural prompt (first 64 streamed chunks), for an eyeball + diff
  python3 - >> $R/ttft_$V.txt 2>&1 <<'PY'
import json, urllib.request
body = {"text": "<|im_start|>user\nExplain in detail how a hash table handles collisions.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "sampling_params": {"temperature": 0}, "stream": True}
req = urllib.request.Request("http://127.0.0.1:30200/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
last, n = "", 0
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line:
            last = json.loads(line[5:])["text"]; n += 1
            if n >= 120: break
print("SAMPLE", json.dumps(last[:600]))
PY
  docker rm -f ft-k08 >/dev/null 2>&1; wait $SP
done
echo K08AB_DONE >> $R/done.txt
