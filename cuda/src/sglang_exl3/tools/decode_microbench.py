"""Where does a decode step's linear time go? Per shape class of a checkpoint, GPU time per call under a
CUDA graph (so launch overhead is excluded, as in vLLM's full graphs), times the number of such linears.

    python -m sglang_exl3.tools.decode_microbench <checkpoint dir> --rows 1 8

Variants: kernel = exl3_gemm as the plugin calls it (extension decides: int8 GEMV / fp16 GEMV / tuned GEMM);
dense = cached decoded fp16 weight: Hadamard-in, cuBLAS, Hadamard-out; floor = a plain fp16 matmul of that shape.
Run once with EXL3_INT8_GEMV=0 in the environment to see the kernel without the int8 shortcut.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

from trellis_core.format import load_manifest
from ..kernels import reference

_CB = {"3inst": 0, "mcg": 1, "mul1": 2}


def graph_time_ms(fn, iters=200) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--skip", nargs="*", default=["visual", "mtp."], help="key substrings not part of a decode step")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    man = load_manifest(args.model_dir)
    classes = defaultdict(list)
    for key, m in man.matrices.items():
        if not any(s in key for s in args.skip):
            classes[(m.k, m.n, m.bits.value, m.codebook.value)].append(key)
    ext = reference._load()
    report = {"gpu": torch.cuda.get_device_name(0), "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
              "classes": []}
    totals = defaultdict(float)
    for (k, n, bits, cb), keys in sorted(classes.items()):
        spec = man.matrices[keys[0]]
        t = {}
        for suffix, info in spec.tensors.items():
            with safe_open(os.path.join(args.model_dir, info.file), "pt") as f:
                t[suffix] = f.get_tensor(info.name).cuda()
        trellis, suh, svh = t["trellis"].contiguous(), t["suh"], t["svh"]
        w = reference.reconstruct(trellis, _CB[cb])
        entry = {"k": k, "n": n, "K": bits, "codebook": cb, "count": len(keys), "example": keys[0], "ms": {}}
        for rows in args.rows:
            x = torch.randn((rows, k), dtype=torch.float16, device="cuda") * 0.5
            xh, y = torch.empty_like(x), torch.empty((rows, n), dtype=torch.float16, device="cuda")

            def kernel():
                ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, cb == "mcg", cb == "mul1", 0)

            def dense():
                ext.had_r_128(x, xh, suh, None, 1.0)
                torch.mm(xh, w, out=y)
                ext.had_r_128(y, y, None, svh, 1.0)

            def floor():
                torch.mm(x, w, out=y)

            ms = {name: graph_time_ms(fn) for name, fn in (("kernel", kernel), ("dense", dense), ("floor", floor))}
            entry["ms"][rows] = ms
            for name, v in ms.items():
                totals[(rows, name)] += v * len(keys)
        report["classes"].append(entry)
        del w, trellis
        line = "  ".join(f"rows={r}: kernel {m['kernel']*1e3:6.0f}us dense {m['dense']*1e3:6.0f}us floor {m['floor']*1e3:6.0f}us"
                         for r, m in entry["ms"].items())
        print(f"{k:>6}->{n:<6} K={bits:g} x{len(keys):<4} {line}", flush=True)
    report["totals_ms_per_step"] = {f"rows={r} {name}": round(v, 3) for (r, name), v in sorted(totals.items())}
    print("\nlinear time per decode step (ms):")
    for label, v in report["totals_ms_per_step"].items():
        print(f"  {label:<18} {v:8.2f}")
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
