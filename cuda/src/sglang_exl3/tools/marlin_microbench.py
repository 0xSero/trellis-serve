"""Per-call GPU time of the Marlin EXL3 kernel next to ExLlamaV3's (and optionally Marlin), CUDA-graph replay.

    PYTHONPATH=src:csrc/build/lib \
        python -m sglang_exl3.tools.marlin_microbench /models/<checkpoint> --rows 1 2 4 8 16 [--marlin]

Columns (us per call): exl3 = exllamav3_ext.exl3_gemm as the plugin calls it (its own dispatch: int8 GEMV at
rows <= 2, fp16 GEMV to 8 rows, tuned GEMM above); marlin = full linear of our kernel (both Hadamards included);
gemm = our GEMM launch alone (rotated basis); marlin = vLLM AWQ-Marlin g32 on random weights of the same shape.
Per-step totals multiply by the number of such linears in the checkpoint (K=3/4/6 classes the kernel supports; the
others are listed and counted with ExLlamaV3's time in both totals).

`--step` measures a decode step the way an engine runs it: q/k/v, gate/up and GDN qkv/z of a layer are fused groups
(one launch set per group). Columns: exl3-sep = separate exl3_gemm calls; exl3-best = min(separate, one sliced
exl3_mgemm launch) = what our vLLM plugin does today; marlin-sep; marlin-fused = our shard-map launch (3 launches
per group); marlin-fused = Marlin on one (k, sum n) matrix, as vLLM runs AWQ.
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
from ..kernels import marlin, reference
from .decode_microbench import _CB, graph_time_ms


def _marlin_fn(k, n, x, group=32):
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
    from vllm.scalar_type import scalar_types
    w = torch.randn((k, n), dtype=torch.float16, device="cuda")
    _, q, scales, zp = awq_marlin_quantize(w, scalar_types.uint4, group)
    ws, empty = mu.marlin_make_workspace_new(w.device), mu.marlin_make_empty_g_idx(w.device)
    del w
    return lambda: mu.apply_gptq_marlin_linear(
        input=x, weight=q, weight_scale=scales, weight_zp=zp, g_idx=empty, g_idx_sort_indices=empty, workspace=ws,
        wtype=scalar_types.uint4, output_size_per_partition=n, input_size_per_partition=k, is_k_full=True)


GROUPS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj"),
          ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"))


def _load(model_dir, spec):
    t = {}
    for suffix, info in spec.tensors.items():
        with safe_open(os.path.join(model_dir, info.file), "pt") as f:
            t[suffix] = f.get_tensor(info.name).cuda()
    t["trellis"] = t["trellis"].contiguous()
    return t


def step_bench(args, man) -> int:
    ext = reference._load()
    keys = [k for k in man.matrices if not any(s in k for s in args.skip)]
    grouped, units = set(), defaultdict(list)          # unit class -> list of key tuples
    for key in keys:
        for grp in GROUPS:
            if key.endswith(grp[0]):
                gkeys = tuple(key[: -len(grp[0])] + g for g in grp)
                specs = [man.matrices.get(g) for g in gkeys]
                if (all(s is not None and s.bits.value in (3, 4, 5) and s.k % 128 == 0 and s.n % 128 == 0 for s in specs)   # K3, K5
                        and len({s.bits.value for s in specs}) == 1):  # K=6: lm_head only; one K per fused group
                    units[("group", specs[0].k, tuple(s.n for s in specs), specs[0].codebook.value)].append(gkeys)
                    grouped.update(gkeys)
    for key in keys:
        if key not in grouped:
            m = man.matrices[key]
            units[("single", m.k, (m.n,), m.codebook.value, m.bits.value)].append((key,))
    names = ["exl3-sep", "exl3-best", "marlin-sep", "marlin-fused", "marlin-bf16"] + (["marlin-fused"] if args.marlin else [])
    totals = defaultdict(float)
    report = {"gpu": torch.cuda.get_device_name(0), "units": []}
    print(f"{'unit':<34}{'count':>5}  " + "  ".join(f"rows={r}: " + " / ".join(names) for r in args.rows[:1]) + "  (us per unit)")
    for ukey, insts in sorted(units.items(), key=lambda kv: str(kv[0])):
        kind, k, ns, cbname = ukey[:4]
        specs = [man.matrices[x] for x in insts[len(insts) // 2]]
        ts = [_load(args.model_dir, s) for s in specs]
        cb = _CB[cbname]
        ok = all(marlin.supports_matrix(t["trellis"])[0] for t in ts)
        singles = [marlin.prepare_matrix(t["trellis"]) for t in ts] if ok else None
        if ok and kind == "group":
            packed, suh_cat, svh_cat, ends = marlin.prepare([t["trellis"] for t in ts], [t["suh"] for t in ts],
                                                                  [t["svh"] for t in ts])
        entry = {"kind": kind, "k": k, "n": list(ns), "count": len(insts), "us": {}}
        cells = []
        for rows in args.rows:
            x = torch.randn((rows, k), dtype=torch.float16, device="cuda") * 0.5
            xh = torch.empty((len(ts) * rows, k), dtype=torch.float16, device="cuda")
            xh1 = xh[:rows]
            ys = [torch.empty((rows, n), dtype=torch.float16, device="cuda") for n in ns]
            ycat = torch.empty((rows, sum(ns)), dtype=torch.float16, device="cuda")
            fns = {"exl3-sep": lambda: [ext.exl3_gemm(x, t["trellis"], y, t["suh"], xh1, t["svh"], -1, cbname == "mcg",
                                                       cbname == "mul1", 0) for t, y in zip(ts, ys)]}
            if kind == "group":
                sg = reference.SlicedGroup([t["trellis"] for t in ts], [t["suh"] for t in ts], [t["svh"] for t in ts], cb, rows)
                fns["exl3-sliced"] = lambda: sg.run(x)
            if ok:
                fns["marlin-sep"] = lambda: [marlin.linear(x, p, t["suh"], t["svh"], cb, xh1, y) for p, t, y in zip(singles, ts, ys)]
                xb, ybs, ybcat = x.to(torch.bfloat16), [y.to(torch.bfloat16) for y in ys], ycat.to(torch.bfloat16)
                if kind == "group":
                    fns["marlin-fused"] = lambda: marlin.linear_group(x, packed, suh_cat, svh_cat, ends, cb, xh, ycat)
                    fns["marlin-bf16"] = lambda: marlin.linear_group(xb, packed, suh_cat, svh_cat, ends, cb, xh, ybcat)
                else:   # bf16 in, bf16 out (what the plugin runs): same launches, conversions inside the Hadamards
                    fns["marlin-bf16"] = lambda: [marlin.linear(xb, p, t["suh"], t["svh"], cb, xh1, y) for p, t, y in zip(singles, ts, ybs)]
            if args.marlin and sum(ns) <= 65536:
                fns["marlin-fused"] = _marlin_fn(k, sum(ns), x)
            us = {name: graph_time_ms(fn, args.iters) * 1e3 for name, fn in fns.items()}
            us["exl3-best"] = min(us["exl3-sep"], us.get("exl3-sliced", 1e9))
            us.setdefault("marlin-sep", us["exl3-best"])
            us.setdefault("marlin-fused", us["marlin-sep"])
            us.setdefault("marlin-bf16", us["marlin-fused"])
            entry["us"][rows] = us
            for name in names:
                totals[(rows, name)] += us.get(name, us["exl3-best"]) * len(insts) / 1e3
            cells.append(f"r{rows}: " + "/".join(f"{us[nm]:.1f}" if nm in us else "-" for nm in names))
        report["units"].append(entry)
        print(f"{kind:<6} {k:>6}->{'+'.join(map(str, ns)):<20}{len(insts):>5}  " + "   ".join(cells)
              + ("" if ok else "   [marlin n/a: exl3 time used]"), flush=True)
    report["totals_ms_per_step"] = {f"rows={r} {name}": round(v, 3) for (r, name), v in sorted(totals.items())}
    print("\nlinear ms per decode step:")
    for r in args.rows:
        print(f"  rows={r:<3} " + "  ".join(f"{name} {totals[(r, name)]:7.2f}" for name in names))
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--skip", nargs="*", default=["visual", "mtp."])
    ap.add_argument("--marlin", action="store_true")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--step", action="store_true", help="whole decode step with fused groups (see module docstring)")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    man = load_manifest(args.model_dir)
    if args.step:
        return step_bench(args, man)
    classes = defaultdict(list)
    for key, m in man.matrices.items():
        if not any(s in key for s in args.skip):
            classes[(m.k, m.n, m.bits.value, m.codebook.value)].append(key)
    ext = reference._load()
    report = {"gpu": torch.cuda.get_device_name(0), "classes": []}
    totals = defaultdict(float)
    names = ["exl3", "marlin", "hop-bf16", "gemm"] + (["marlin"] if args.marlin else [])
    print(f"{'shape':<16}{'K':>3} {'count':>5}  " + "  ".join(f"rows={r:<3}" + "/".join(names) for r in args.rows))
    for (k, n, bits, cb), keys in sorted(classes.items()):
        spec = man.matrices[keys[0]]
        t = {}
        for suffix, info in spec.tensors.items():
            with safe_open(os.path.join(args.model_dir, info.file), "pt") as f:
                t[suffix] = f.get_tensor(info.name).cuda()
        trellis, suh, svh = t["trellis"].contiguous(), t["suh"], t["svh"]
        ok, why = marlin.supports_matrix(trellis)
        packed = marlin.prepare_matrix(trellis) if ok else None
        entry = {"k": k, "n": n, "K": bits, "codebook": cb, "count": len(keys), "marlin": ok, "why": why, "us": {}}
        cells = []
        for rows in args.rows:
            x = torch.randn((rows, k), dtype=torch.float16, device="cuda") * 0.5
            xh, y = torch.empty_like(x), torch.empty((rows, n), dtype=torch.float16, device="cuda")
            fns = {"exl3": lambda: ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, cb == "mcg", cb == "mul1", 0)}
            if ok:
                fns["marlin"] = lambda: marlin.linear(x, packed, suh, svh, _CB[cb], xh, y)
                xb, yb = x.to(torch.bfloat16), y.to(torch.bfloat16)
                fns["hop-bf16"] = lambda: marlin.linear(xb, packed, suh, svh, _CB[cb], xh, yb)
                fns["gemm"] = lambda: marlin.gemm_rotated(x, packed, _CB[cb], y)
            if args.marlin and n <= 65536:
                fns["marlin"] = _marlin_fn(k, n, x)
            us = {name: graph_time_ms(fn, args.iters) * 1e3 for name, fn in fns.items()}
            entry["us"][rows] = us
            for name in names:
                totals[(rows, name)] += us.get(name, us["exl3"]) * len(keys) / 1e3
            cells.append("/".join(f"{us[nm]:5.1f}" if nm in us else "    -" for nm in names))
        report["classes"].append(entry)
        print(f"{k:>6}->{n:<8}{bits:>3g} {len(keys):>5}  " + "  ".join(f"{c:<{8 + 6 * len(names)}}" for c in cells)
              + ("" if ok else f"   [marlin: {why}]"), flush=True)
        del trellis, packed
    report["totals_ms_per_step"] = {f"rows={r} {name}": round(v, 3) for (r, name), v in sorted(totals.items())}
    print("\nlinear ms per decode step (unsupported classes counted at ExLlamaV3's time):")
    for r in args.rows:
        print(f"  rows={r:<3} " + "  ".join(f"{name} {totals[(r, name)]:7.2f}" for name in names))
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
