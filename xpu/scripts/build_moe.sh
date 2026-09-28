#!/usr/bin/env bash
# Build exl3xpu/_moe.so (grouped EXL3 MoE kernels, torch library exl3xpu_moe). Same toolchain as build_ext.sh.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared \
  -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${EXL3_MOE_FLAGS:-} \
  -I csrc -I$T/include -I$T/include/torch/csrc/api/include \
  -x c++ csrc/exl3_moe.sycl -x none -o ${EXL3_MOE_OUT:-exl3xpu/_moe.so} \
  -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu 2>&1 | grep -E "error|Error" -A3 || true
ls -la ${EXL3_MOE_OUT:-exl3xpu/_moe.so}
