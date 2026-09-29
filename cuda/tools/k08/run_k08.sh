#!/bin/bash
# K08: serve on GPU 0 (integration serve script), profile one 32k prompt's 8k prefill chunks, stop the server.
# Run under bench/cpu_run.sh (server start + prefill are CPU/DRAM heavy).
set -u
cd ~/freetoken-exl3
R=${1:-runs/2026-09-29-K08-prefill-profile}; mkdir -p $R/prof
export GPU=0 PORT=30200 NAME=ft-k08 CACHE_GB=${CACHE_GB:-6} MEMFRAC=${MEMFRAC:-0.80}
integration/trellis-serve/cuda/scripts/serve_flashnext_3090.sh $R/serve &
SP=$!
for i in $(seq 1 180); do
  curl -s -m 5 http://127.0.0.1:30200/health >/dev/null 2>&1 && break
  grep -q "^exit" $R/serve/log.txt 2>/dev/null && break
  sleep 5
done
curl -s -m 5 http://127.0.0.1:30200/health && echo " server up" || { echo "server not up"; tail -30 $R/serve/log.txt; docker rm -f ft-k08; exit 1; }
python3 kernels/cuda_sm86/k08/profile_prefill.py --url http://127.0.0.1:30200 --ctx 4096 --steps 1 --dir /w/$R/warm --host-dir $R/warm --no-profile
for rep in 1 2; do python3 kernels/cuda_sm86/k08/profile_prefill.py --url http://127.0.0.1:30200 --ctx 32768 --no-profile --dir x --host-dir x; done > $R/ttft.txt 2>&1
python3 kernels/cuda_sm86/k08/profile_prefill.py --url http://127.0.0.1:30200 --ctx 32768 --steps 4 --dir /w/$R/prof --host-dir $R/prof > $R/breakdown.txt 2>&1
docker rm -f ft-k08 >/dev/null 2>&1
wait $SP
echo K08_DONE >> $R/breakdown.txt
