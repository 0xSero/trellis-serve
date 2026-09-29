"""K06a: 8-bit HC mix vs SGLang's bf16 persistent kernel on the REAL hyper-connection weights (96 sites + final mixer).

    python -m sglang_exl3.tools.hc_int8_bench /models/<Flash-Next> [--rows 1,4,16] [--json out.json]

Activations: x = grouped RMSNorm (per 2560-wide stream, Gemma-style (1 + w) with the site's REAL hc_norm weight) of
random streams whose per-stream scales differ (log-normal) -- the kernel only ever sees normed input.
Reference: fp32 math with the checkpoint's fp16 weights. Error metric per site: ||y - y_ref|| / ||y_ref|| (rel L2) and
max |y - y_ref|; reported as the mean / max over sites, for SGLang's bf16 kernel (the current baseline), int8 per-row
and int8 per-group-32. Timing: one CUDA graph of all 97 sites (distinct weights, so nothing stays in L2), median of 50.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
from safetensors import safe_open

from ..kernels import hc_mix_int8 as hq

HC, HS = 4, 2560


def load_sites(model):
    wm = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    prefixes = sorted({k.rsplit(".input_mix_weight_down", 1)[0] for k in wm if k.endswith("input_mix_weight_down.weight")
                       and not k.startswith("mtp.")})
    handles = {}

    def get(k):
        f = wm[k]
        if f not in handles:
            handles[f] = safe_open(os.path.join(model, f), "pt")
        return handles[f].get_tensor(k)

    sites = []
    for p in prefixes:
        sites.append({"name": p, "down": get(p + ".input_mix_weight_down.weight").cuda(),
                      "up": get(p + ".input_mix_weight_up.weight").cuda(), "norm": get(p + ".hc_norm.weight").float().cuda()})
    return sites


def normed_input(rows, norm_w, gen):
    s = torch.exp(torch.randn((rows, HC, 1), generator=gen) * 0.7)
    x = (torch.randn((rows, HC, HS), generator=gen) * s).cuda()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (x.flatten(-2) * (1.0 + norm_w)).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--rows", default="1,4,16"); ap.add_argument("--json")
    a = ap.parse_args()
    from sglang.srt.layers.hc_mix_triton import fused_hc_mix
    sites = load_sites(a.model)
    print("sites", len(sites), "dtype", sites[0]["down"].dtype, tuple(sites[0]["down"].shape), tuple(sites[0]["up"].shape), flush=True)
    bf = [(s["down"].to(torch.bfloat16).contiguous(), s["up"].to(torch.bfloat16).contiguous()) for s in sites]
    q0 = [hq.HcInt8Weights(d, u, 0, 0) for d, u in bf]
    q32 = [hq.HcInt8Weights(d, u, 128, 64) for d, u in bf]
    bytes_bf = sum(d.numel() * 2 + u.numel() * 2 for d, u in bf)
    out = {"sites": len(sites), "GB_bf16": bytes_bf / 1e9, "GB_int8_row": sum(q.nbytes() for q in q0) / 1e9,
           "GB_int8_g128_64": sum(q.nbytes() for q in q32) / 1e9, "accuracy": {}, "timing_us": {}}
    gen = torch.Generator().manual_seed(0)
    for rows in [int(r) for r in a.rows.split(",")]:
        errs = {"bf16_sglang": [], "int8_row": [], "int8_g128_64": [], "int8_row_torch_vs_kernel": []}
        mx = {k: 0.0 for k in errs}
        for i, s in enumerate(sites):
            x = normed_input(rows, s["norm"], gen)
            ref = hq.hc_mix_reference(x.float(), s["down"].float(), s["up"].float(), HC, HS)
            ys = {"bf16_sglang": fused_hc_mix(x, bf[i][0], bf[i][1], HC, HS).float(),
                  "int8_row": hq.fused_hc_mix_int8(x, q0[i], HC, HS).float(),
                  "int8_g128_64": hq.fused_hc_mix_int8(x, q32[i], HC, HS).float()}
            for k, y in ys.items():
                e = (y - ref).norm() / ref.norm()
                errs[k].append(float(e)); mx[k] = max(mx[k], float((y - ref).abs().max()))
            yt = hq.hc_mix_int8_torch(x, q0[i], HC, HS).float()
            errs["int8_row_torch_vs_kernel"].append(float((yt - ys["int8_row"]).norm() / ys["int8_row"].norm()))
        out["accuracy"][rows] = {k: {"rel_l2_mean": statistics.mean(v), "rel_l2_max": max(v), "max_abs": mx[k]}
                                 for k, v in errs.items()}
        print("ACC rows", rows, json.dumps(out["accuracy"][rows]), flush=True)
    # timing: all sites in one graph
    for rows in [int(r) for r in a.rows.split(",")]:
        xs = [normed_input(rows, s["norm"], gen) for s in sites]
        variants = {"bf16_sglang": lambda: [fused_hc_mix(x, d, u, HC, HS) for x, (d, u) in zip(xs, bf)],
                    "int8_row": lambda: [hq.fused_hc_mix_int8(x, q, HC, HS) for x, q in zip(xs, q0)],
                    "int8_g128_64": lambda: [hq.fused_hc_mix_int8(x, q, HC, HS) for x, q in zip(xs, q32)]}
        res = {}
        for name, fn in variants.items():
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                fn(); fn()
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            ts = []
            for r in range(55):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                if r >= 5:
                    ts.append(e0.elapsed_time(e1) * 1000)
            res[name] = statistics.median(ts)
        out["timing_us"][rows] = res
        print("TIME rows", rows, {k: round(v, 1) for k, v in res.items()}, "(97 sites, us)", flush=True)
    # tile sweep (rows = 1) for both int8 variants
    xs = [normed_input(1, s["norm"], gen) for s in sites]
    sweep = []
    for bn in (32, 64):
        for bk in (256, 512):
            for bj in (16, 32, 64):
                for nw in (4, 8):
                    cfg = {"BLOCK_N": bn, "BLOCK_K": bk, "BLOCK_J": bj, "BLOCK_R": 64, "num_warps": nw}
                    for vname, qs in (("int8_row", q0), ("int8_g128_64", q32)):
                        fn = lambda: [hq.fused_hc_mix_int8(x, q, HC, HS, cfg) for x, q in zip(xs, qs)]
                        try:
                            fn(); torch.cuda.synchronize()
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                fn()
                            ts = []
                            for r in range(25):
                                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                                if r >= 5:
                                    ts.append(e0.elapsed_time(e1) * 1000)
                            sweep.append((round(statistics.median(ts), 1), vname, cfg))
                        except Exception as ex:
                            sweep.append((1e9, vname, cfg, str(ex)[:80]))
    sweep.sort(key=lambda t: t[0])
    out["sweep"] = sweep
    for v in ("int8_row", "int8_g128_64"):
        best = [t for t in sweep if t[1] == v][:3]
        print("SWEEP", v, best, flush=True)
    print("HC_INT8", json.dumps({k: out[k] for k in ("sites", "GB_bf16", "GB_int8_row", "GB_int8_g128_64")}), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
