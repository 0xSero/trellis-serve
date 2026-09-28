"""
X001: MUL1 K=3 on XPU, validated on REAL Qwen3.8-Flash-Next routed-expert tensors.

1. exl3_reconstruct (ESIMD) vs trellis_core.reference.reconstruct_inner: bit-exact, every expert projection sampled.
2. exl3_gemm_raw one-hot rows (vector M=1/2/4, DPAS M=8..64): every fp32 output an exact copy of one decoded
   weight -> bit-exact vs the reference decode.
3. exl3_gemm_small (whole linear incl. both Hadamards) vs reference.linear_forward (fp32): fp16 tolerance.
4. Microbench: dense GEMV/GEMM for 2560x640 (gate/up) and 640x2560 (down), M = 1..16.

Usage (in the exl3xpu container): python3 tests/test_k3_experts.py [--layers 0,23,47] [--experts 8] [--bench]
"""
import os, sys, json, time, random, argparse
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core", "src"))
from trellis_core import reference as ref

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "/model"))
ap.add_argument("--lib", default=os.environ.get("EXL3_LIB", os.path.join(HERE, "..", "exl3xpu", "_C.so")))
ap.add_argument("--layers", default="0,1,23,46,47")
ap.add_argument("--experts", type=int, default=6)
ap.add_argument("--bench", action="store_true")
ap.add_argument("--dense-regex", default="", help="validate these (non-expert) tensors instead, any K")
ap.add_argument("--out", default="")
args = ap.parse_args()

torch.ops.load_library(args.lib)
E = torch.ops.exl3xpu_C
dev = torch.device("xpu:0")
idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
handles = {}


def get(name):
    fn = idx[name]
    if fn not in handles:
        handles[fn] = safe_open(f"{args.model}/{fn}", "pt", device="cpu")
    return handles[fn].get_tensor(name)


res = {"reconstruct": [], "onehot": [], "linear": [], "bench": []}
fails = []
rng = random.Random(0)
layers = [int(x) for x in args.layers.split(",")]
t0 = time.time()
import re as _re


def work():
    if args.dense_regex:
        rx = _re.compile(args.dense_regex)
        for name in sorted(idx):
            if name.endswith(".trellis") and rx.search(name):
                yield None, name[:-len(".trellis")]
        return
    for L in layers:
        for e in sorted(rng.sample(range(512), args.experts)):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield L, f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}"


lastL = None
for L, key in work():
            if L != lastL and lastL is not None:
                print(f"layer {lastL} done ({time.time() - t0:.1f}s), fails so far {len(fails)}", flush=True)
            lastL = L
            assert f"{key}.mul1" in idx, key
            tr = get(f"{key}.trellis").to(dev)
            suh, svh = get(f"{key}.suh").to(dev), get(f"{key}.svh").to(dev)
            K = tr.shape[-1] // 16
            assert K == 3 or args.dense_regex, (key, K)
            res.setdefault("K_seen", {}).setdefault(str(K), 0)
            res["K_seen"][str(K)] += 1
            cb = ref.CB_MUL1
            k, n = tr.shape[0] * 16, tr.shape[1] * 16
            wr = ref.reconstruct_inner(tr, K, cb)
            wx = torch.empty((k, n), dtype=torch.float16, device=dev)
            E.exl3_reconstruct(tr, wx, 0, K, cb)
            bad = (wx.view(torch.int16) != wr.view(torch.int16)).sum().item()
            res["reconstruct"].append(bad)
            if bad:
                fails.append(f"reconstruct {key}: {bad}")
            # one-hot GEMM paths (sampled k rows)
            shard = torch.zeros(n // 128, dtype=torch.int32, device=dev)
            rows = rng.sample(range(k), 64)
            for path, M in [(0, 1), (0, 2), (0, 4), (1, 8), (1, 16), (1, 32), (1, 64)]:
                nb = 0
                for c in range(0, len(rows), M):
                    ks = rows[c:c + M]
                    xh = torch.zeros((1, k // 16, len(ks), 16), dtype=torch.float16, device=dev)
                    for m, kk in enumerate(ks):
                        xh[0, kk // 16, m, kk % 16] = 1.0
                    part = E.exl3_gemm_raw(xh, tr, shard, n, K, cb, path)
                    nb += (part != wr[ks].float()).sum().item()
                res["onehot"].append(nb)
                if nb:
                    fails.append(f"onehot {key} path={path} M={M}: {nb}")
            # whole linear vs fp32 reference
            for M in (1, 2, 3, 8, 16, 33):
                x = torch.randn((M, k), dtype=torch.float16, device=dev)
                out = torch.empty((M, n), dtype=torch.float16, device=dev)
                E.exl3_gemm_small(x, tr, suh.view(1, -1), svh, shard, out, K, cb)
                y = ref.linear_forward(x, tr, suh, svh, K, cb)
                err = ((out.float() - y).abs().max() / y.abs().max()).item()
                res["linear"].append(err)
                if err > 2e-3:
                    fails.append(f"linear {key} M={M}: rel max err {err:.2e}")
print(f"done ({time.time() - t0:.1f}s), fails {len(fails)}, K seen {res.get('K_seen')}", flush=True)

print("reconstruct tensors:", len(res["reconstruct"]), "mismatching elems:", sum(res["reconstruct"]))
print("onehot cases:", len(res["onehot"]), "mismatching:", sum(res["onehot"]))
print("linear cases:", len(res["linear"]), "max rel err:", max(res["linear"]))

if args.bench:
    def timeit(fn, it=200):
        for _ in range(10):
            fn()
        torch.xpu.synchronize()
        t = time.perf_counter()
        for _ in range(it):
            fn()
        torch.xpu.synchronize()
        return (time.perf_counter() - t) / it * 1e6

    L, e = 0, 0
    for proj in ("gate_proj", "down_proj"):
        key = f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}"
        tr = get(f"{key}.trellis").to(dev)
        suh, svh = get(f"{key}.suh").to(dev).view(1, -1), get(f"{key}.svh").to(dev)
        k, n = tr.shape[0] * 16, tr.shape[1] * 16
        shard = torch.zeros(n // 128, dtype=torch.int32, device=dev)
        for M in (1, 2, 4, 8, 16):
            x = torch.randn((M, k), dtype=torch.float16, device=dev)
            out = torch.empty((M, n), dtype=torch.float16, device=dev)
            us = timeit(lambda: E.exl3_gemm_small(x, tr, suh, svh, shard, out, 3, 2))
            gbs = tr.numel() * 2 / us / 1e3
            r = {"shape": f"{k}x{n}", "M": M, "us": round(us, 2), "weight_GBps": round(gbs, 1)}
            res["bench"].append(r)
            print(r, flush=True)
    # 10 experts back to back (what a naive per-expert decode would do): gate+up+down per expert
    trs = []
    for e in range(10):
        ts = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}"
            ts[proj] = (get(f"{key}.trellis").to(dev), get(f"{key}.suh").to(dev).view(1, -1), get(f"{key}.svh").to(dev))
        trs.append(ts)
    s640 = torch.zeros(5, dtype=torch.int32, device=dev)
    s2560 = torch.zeros(20, dtype=torch.int32, device=dev)
    x = torch.randn((1, 2560), dtype=torch.float16, device=dev)
    g = torch.empty((1, 640), dtype=torch.float16, device=dev)
    u = torch.empty((1, 640), dtype=torch.float16, device=dev)
    d = torch.empty((1, 2560), dtype=torch.float16, device=dev)

    def naive():
        for ts in trs:
            E.exl3_gemm_small(x, *ts["gate_proj"][:1], ts["gate_proj"][1], ts["gate_proj"][2], s640, g, 3, 2)
            E.exl3_gemm_small(x, *ts["up_proj"][:1], ts["up_proj"][1], ts["up_proj"][2], s640, u, 3, 2)
            a = torch.nn.functional.silu(g) * u
            E.exl3_gemm_small(a, *ts["down_proj"][:1], ts["down_proj"][1], ts["down_proj"][2], s2560, d, 3, 2)

    us = timeit(naive, 50)
    r = {"case": "naive 10 experts x (gate, up, silu*mul, down), M=1", "us": round(us, 1),
         "weight_GBps": round(10 * 1843200 / us / 1e3, 1)}
    res["bench"].append(r)
    print(r)

if args.out:
    json.dump({"fails": fails, **{k: v for k, v in res.items() if k != "linear"},
               "linear_max_rel_err": max(res["linear"])}, open(args.out, "w"), indent=1)
print("FAILS:", len(fails))
for f in fails[:20]:
    print("  ", f)
sys.exit(1 if fails else 0)
