"""
X002: grouped EXL3 MoE (pointer-table experts) on XPU vs the trellis_core fp32 reference, on REAL Qwen3.8-Flash-Next
routed experts; plus decode/prefill microbenchmarks with experts in device slots / host USM (zero-copy) / mixed.

  python3 tests/test_moe.py --check            # correctness, layers 0 and 47, M = 1..16 (+ prefill sizes if built)
  python3 tests/test_moe.py --bench-decode     # per-layer latency, device vs host-USM vs mixed
"""
import os, sys, json, time, argparse, random
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))
from trellis_core import reference as ref

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "/model"))
ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--check", action="store_true")
ap.add_argument("--check-m", default="1,2,3,4,8,16")
ap.add_argument("--bench-decode", action="store_true")
ap.add_argument("--bench-m", default="1,2,4,8,16")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--out", default="")
args = ap.parse_args()

torch.ops.load_library(os.path.join(HERE, "..", "exl3xpu", "_moe.so"))
X = torch.ops.exl3xpu_moe
dev = torch.device("xpu:0")
H, I, K, E, TOPK = 2560, 640, 3, 512, 10
BLOB = X.blob_bytes(H, I, K)
dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
handles = {}


def get(name):
    fn = idx[name]
    if fn not in handles:
        handles[fn] = safe_open(f"{args.model}/{fn}", "pt", device="cpu")
    return handles[fn].get_tensor(name)


def expert_tensors(L, e):
    p = f"model.language_model.layers.{L}.mlp.experts.{e}."
    return {pr: {s: get(p + pr + "." + s) for s in ("trellis", "suh", "svh")} for pr in ("gate_proj", "up_proj", "down_proj")}


def pack(t):
    b = lambda z: z.contiguous().view(torch.uint8).flatten()
    gu = torch.cat([t["gate_proj"]["trellis"], t["up_proj"]["trellis"]], dim=1)
    blob = torch.cat([b(gu), b(t["down_proj"]["trellis"]), b(t["gate_proj"]["suh"]), b(t["up_proj"]["suh"]),
                      b(t["gate_proj"]["svh"]), b(t["up_proj"]["svh"]), b(t["down_proj"]["suh"]), b(t["down_proj"]["svh"])])
    assert blob.numel() == BLOB, (blob.numel(), BLOB)
    return blob


def s64(p):
    return p - (1 << 64) if p >= (1 << 63) else p


def load_layer(L):
    t0 = time.time()
    host = X.host_alloc(E * BLOB).view(E, BLOB)          # USM host (zero-copy readable by the device)
    for e in range(E):
        host[e].copy_(pack(expert_tensors(L, e)))
    devarena = host.to(dev)                               # device slots [E, BLOB]
    print(f"layer {L}: packed {E} experts ({E * BLOB / 1e9:.3f} GB) in {time.time() - t0:.1f}s; "
          f"usm_kind host={X.usm_kind(s64(host.data_ptr()))} dev={X.usm_kind(s64(devarena.data_ptr()))}", flush=True)
    return host, devarena


def ptr_table(host, devarena, on_host_mask):
    hp = s64(host.data_ptr()) + torch.arange(E, dtype=torch.int64) * BLOB
    dp = s64(devarena.data_ptr()) + torch.arange(E, dtype=torch.int64) * BLOB
    return torch.where(on_host_mask, hp, dp).to(dev)


def ref_moe(x, ids, w, L, cache):
    """fp32 reference: per (token, expert) gate/up/down via trellis_core.reference.linear_forward."""
    out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=dev)
    for m in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[m, k])
            if e not in cache:
                cache[e] = {pr: {s: v.to(dev) for s, v in d.items()} for pr, d in expert_tensors(L, e).items()}
            t = cache[e]
            xm = x[m:m + 1].float()
            g = ref.linear_forward(xm, t["gate_proj"]["trellis"], t["gate_proj"]["suh"], t["gate_proj"]["svh"], K, 2)
            u = ref.linear_forward(xm, t["up_proj"]["trellis"], t["up_proj"]["suh"], t["up_proj"]["svh"], K, 2)
            a = F.silu(g) * u
            d = ref.linear_forward(a, t["down_proj"]["trellis"], t["down_proj"]["suh"], t["down_proj"]["svh"], K, 2)
            out[m] += float(w[m, k]) * d[0]
    return out


def routing(M, g=None, skew=None):
    g = g or torch.Generator().manual_seed(0)
    logits = torch.randn((M, E), generator=g)
    if skew is not None:
        logits = logits + skew
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return ids.to(torch.int32).to(dev), w.float().to(dev)


res = {"check": [], "decode": []}
host, devarena = load_layer(args.layer)

if args.check:
    cache = {}
    torch.manual_seed(0)
    for M in [int(v) for v in args.check_m.split(",")]:
        for placement in ("device", "host", "mixed"):
            mask = {"device": torch.zeros(E, dtype=torch.bool), "host": torch.ones(E, dtype=torch.bool),
                    "mixed": torch.arange(E) % 2 == 1}[placement]
            ptrs = ptr_table(host, devarena, mask)
            x = (torch.randn((M, H), device=dev) * 0.5).to(dt)
            ids, w = routing(M, torch.Generator().manual_seed(M))
            y = X.moe_forward(x, ids, w, ptrs, I, K, E)
            y2 = X.moe_forward(x, ids, w, ptrs, I, K, E)
            torch.xpu.synchronize()
            r = ref_moe(x, ids.cpu(), w.cpu(), args.layer, cache)
            err = ((y.float() - r).abs().max() / r.abs().max()).item()
            rms = ((y.float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
            det = torch.equal(y, y2)
            floor = ((r.to(dt).float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
            rec = {"M": M, "placement": placement, "max_rel": round(err, 5), "rms_rel": round(rms, 5),
                   "rms_rel_floor_out_dtype": round(floor, 5), "deterministic": det}
            res["check"].append(rec)
            print(rec, flush=True)

if args.bench_decode:
    def bench(fn_list, iters):
        for f in fn_list[:5]:
            f()
        torch.xpu.synchronize()
        t = time.perf_counter()
        for i in range(iters):
            fn_list[i % len(fn_list)]()
        torch.xpu.synchronize()
        return (time.perf_counter() - t) / iters * 1e6

    for M in [int(v) for v in args.bench_m.split(",")]:
        x = (torch.randn((M, H), device=dev) * 0.5).to(dt)
        routs = [routing(M, torch.Generator().manual_seed(1000 + i)) for i in range(64)]
        for placement, frac in (("device", 0.0), ("host", 1.0), ("mixed50", 0.5), ("mixed20", 0.2)):
            mask = torch.rand(E, generator=torch.Generator().manual_seed(7)) < frac
            ptrs = ptr_table(host, devarena, mask)
            fns = [lambda ids=ids, w=w: X.moe_forward(x, ids, w, ptrs, I, K, E) for ids, w in routs]
            us = bench(fns, args.iters)
            n_unique = sum(len(torch.unique(ids)) for ids, _ in routs) / len(routs)
            gb = n_unique * BLOB / 1e9
            rec = {"M": M, "placement": placement, "us_per_layer": round(us, 1), "unique_experts": n_unique,
                   "weight_GBps": round(gb / (us * 1e-6), 1), "ms_48_layers": round(us * 48 / 1e3, 2)}
            res["decode"].append(rec)
            print(rec, flush=True)

if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
