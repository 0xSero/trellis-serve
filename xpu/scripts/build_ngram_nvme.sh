#!/usr/bin/env bash
# Build exl3xpu/exl3xpu_ngram_nvme.so (python extension: NVMe row store + XPU publish/wait) with icpx.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
PY=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
icpx -fsycl -fsycl-targets=spir64 -O3 -fPIC -std=c++17 -shared -march=native -D_GLIBCXX_USE_CXX11_ABI=1 \
  -DTORCH_EXTENSION_NAME=exl3xpu_ngram_nvme -DTORCH_API_INCLUDE_EXTENSION_H \
  -I$T/include -I$T/include/torch/csrc/api/include -I$PY \
  -x c++ csrc/ngram_nvme_xpu.sycl -x none -o ${EXL3_NVME_OUT:-exl3xpu/exl3xpu_ngram_nvme.so} \
  -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu -ltorch_python 2>&1 | grep -E "error" -A3 || true
ls -la ${EXL3_NVME_OUT:-exl3xpu/exl3xpu_ngram_nvme.so}
