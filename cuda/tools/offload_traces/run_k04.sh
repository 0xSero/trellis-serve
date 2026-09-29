#!/bin/bash
# K04 on omarchy GPU 0: exllamav3 CPU worker (-mcl 48, -mct T) + sglang_exl3 offload kernels in one process. CPU-heavy.
T=${1:-16}; R=runs/2026-09-29-K04-split; cd ~/freetoken-exl3; mkdir -p $R
M=/models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
CMD="python3 /w/kernels/cuda_sm86/traces/k04_split.py -m $M -cs 4096 -mcl 48 -mct $T --layer 3 --json /w/$R/k04_t$T.json"
echo "$CMD" >> $R/cmd.txt
docker run --rm --name ft-k04 --gpus device=0 --ipc=host --shm-size=64g --ulimit memlock=-1 -v /home/sero/models:/models \
  -v /home/sero/freetoken-exl3:/w -e PYTHONPATH=/w/kernels/cuda_sm86/trellis-serve/cuda/csrc/build/lib:/w/kernels/cuda_sm86/trellis-serve/cuda/src:/w/kernels/cuda_sm86/trellis-serve/core/src \
  --entrypoint bash sglang-exl3:dev -c "$CMD" >> $R/log_t$T.txt 2>&1; echo "exit $?" >> $R/log_t$T.txt
