"""L0/L1 parity on real checkpoint tensors, one linear at a time (small: runs next to a busy GPU).

Prints one JSON document with content hashes, so the same command run in two environments
(stock ExLlamaV3 wheel vs our patched build, or two GPUs) can be diffed for bit-exactness:

    python -m sglang_exl3.parity.l1_linear <checkpoint dir> <module key> [<module key> ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

from trellis_core.format import load_manifest
from ..kernels import reference

ROWS = (1, 2, 3, 8, 16, 17, 64, 144, 145, 512)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _hadamard128(device) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64, device=device)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 128 ** 0.5


def _block_had(x: torch.Tensor) -> torch.Tensor:
    return (x.view(x.shape[0], -1, 128) @ _hadamard128(x.device)).reshape(x.shape)


def check(model_dir: str, key: str, device="cuda") -> dict:
    spec = load_manifest(model_dir).matrices[key]
    t = {}
    for suffix, info in spec.tensors.items():
        with safe_open(os.path.join(model_dir, info.file), "pt") as f:
            t[suffix] = f.get_tensor(info.name).to(device)
    cb = {"3inst": 0, "mcg": 1, "mul1": 2}[spec.codebook.value]
    trellis, suh, svh = t["trellis"].contiguous(), t["suh"], t["svh"]
    w_hat = reference.reconstruct(trellis, cb)
    out = {"key": key, "k": spec.k, "n": spec.n, "K": spec.bits.value, "codebook": spec.codebook.value,
           "sha_w_hat": _sha(w_hat), "rows": {}}
    w64 = w_hat.double()
    gen = torch.Generator().manual_seed(1234)
    for rows in ROWS:
        x = (torch.randn((rows, spec.k), generator=gen, dtype=torch.float32) * 0.5).to(torch.float16).to(device)
        exact = _block_had(_block_had(x.double() * suh.double()) @ w64) * svh.double()
        y_k = reference.gemm(x, trellis, suh, svh, cb)
        y_k32 = reference.gemm(x, trellis, suh, svh, cb, out_dtype=torch.float32)
        y_d = reference.dense_forward(x, trellis, suh, svh, cb)
        scale = exact.abs().mean().item()
        rel = lambda y: round(((y.double() - exact).abs().mean().item()) / scale, 7)
        out["rows"][rows] = {"sha_kernel_fp16": _sha(y_k), "sha_kernel_fp32": _sha(y_k32), "sha_dense": _sha(y_d),
                             "relerr_kernel_fp16": rel(y_k), "relerr_kernel_fp32": rel(y_k32), "relerr_dense": rel(y_d),
                             "finite": bool(torch.isfinite(y_k).all() and torch.isfinite(y_d).all())}
    return out


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    doc = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
           "backend": reference.probe().__dict__, "linears": [check(argv[0], key) for key in argv[1:]]}
    json.dump(doc, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
