"""JIT build of csrc/offload/offload_kernels.cu (torch cpp_extension, sm_86). Cached under
SGLANG_EXL3_JIT_DIR (default <repo>/cuda/csrc/build/jit), so only the first process on a box compiles."""
from __future__ import annotations

import os
import threading

_mod = None
_lock = threading.Lock()


def load():
    global _mod
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is None:
            from torch.utils import cpp_extension
            here = os.path.dirname(os.path.abspath(__file__))
            csrc = os.path.normpath(os.path.join(here, "..", "..", "..", "csrc"))
            build = os.environ.get("SGLANG_EXL3_JIT_DIR", os.path.join(csrc, "build", "jit"))
            os.makedirs(build, exist_ok=True)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
            _mod = cpp_extension.load(
                name="trellis_offload_kernels",
                sources=[os.path.join(csrc, "offload", "offload_kernels.cu")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
                build_directory=build,
                verbose=False,
            )
    return _mod
