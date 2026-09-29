#!/bin/bash
# K02 trace capture on GPU 0 (docker device=0). CPU-heavy: exllamav3 CPU experts, 16 worker threads.
R=runs/2026-09-29-K02-traces; cd "${CAMPAIGN_DIR:?set CAMPAIGN_DIR}"; mkdir -p $R
M=/models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
CMD="python3 /w/kernels/cuda_sm86/traces/capture_traces.py -m $M -cs 65536 -cq 8 -mcl 48 -mct 16 --workloads /w/kernels/cuda_sm86/traces/workloads.json --out /w/$R $*"
echo "$CMD" >> $R/cmd.txt
docker run --rm --name ft-k02-trace --gpus device=0 --ipc=host --shm-size=64g --ulimit memlock=-1 -v "${MODEL_ROOT:?set MODEL_ROOT}":/models -v "$CAMPAIGN_DIR":/w --entrypoint bash sglang-exl3:dev -c "$CMD" >> $R/log.txt 2>&1; echo "exit $?" >> $R/log.txt
