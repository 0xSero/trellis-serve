"""
Device-managed expert cache (moe_forward_cached): correctness + hit-rate/latency under skewed routing.

Correctness: every cached call must equal (bitwise) the zero-copy call on the host-only table (write-through does
not change the math); after the run every occupied slot must hold exactly its expert's host blob, ptrs must point
at the slot for cached experts and at host memory for the rest.

  python3 tests/test_moe_cache.py --layers 0,1 --slots 256 --steps 200
"""
import os, sys, json, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core", "src"))
from safetensors import safe_open
from exl3xpu.moe_offload import ExpertStore, pack_expert, s64

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "/model"))
ap.add_argument("--layers", default="0,1")
ap.add_argument("--slots", type=int, default=256)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--zipf", type=float, default=1.0, help="expert popularity skew exponent (0 = uniform)")
ap.add_argument("--bench", action="store_true")
ap.add_argument("--out", default="")
args = ap.parse_args()

H, I, K, E, TOPK = 2560, 640, 3, 512, 10
dev = torch.device("xpu", 0)
idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
handles = {}


def get(n):
    fn = idx[n]
    if fn not in handles:
        handles[fn] = safe_open(f"{args.model}/{fn}", "pt", device="cpu")
    return handles[fn].get_tensor(n)


layers = [int(v) for v in args.layers.split(",")]
store = ExpertStore(H, I, K, E, args.slots, dev, max_layers=len(layers))
for L in layers:
    key = f"layers.{L}.mlp.experts"
    h = store.add_layer(key)
    for e in range(E):
        p = f"model.language_model.layers.{L}.mlp.experts.{e}."
        t = {pr: {s: get(p + pr + "." + s) for s in ("trellis", "suh", "svh")} for pr in ("gate_proj", "up_proj", "down_proj")}
        pack_expert(t["gate_proj"], t["up_proj"], t["down_proj"], K, out=h[e])
host_tab = {f"layers.{L}.mlp.experts": store._host_ptr[f"layers.{L}.mlp.experts"].to(dev) for L in layers}
print(f"store: {len(layers)} layers, {args.slots} slots ({args.slots * store.blob / 1e9:.2f} GB)", flush=True)

# per-layer popularity (Zipf over a random permutation)
pop = {}
for L in layers:
    g = torch.Generator().manual_seed(100 + L)
    rank = torch.randperm(E, generator=g)
    pop[L] = -args.zipf * torch.log1p(rank.float())


def route(L, M, step):
    g = torch.Generator().manual_seed(step * 1000 + L)
    logits = torch.randn((M, E), generator=g) + pop[L]
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    return ids.to(torch.int32).to(dev), (w / w.sum(-1, keepdim=True)).float().to(dev)


res = {"steps": [], "bench": []}
bad = 0
misses_total = 0
for step in range(args.steps):
    for L in layers:
        key = f"layers.{L}.mlp.experts"
        M = 1 if step % 3 else 4
        x = (torch.randn((M, H), device=dev) * 0.5).to(torch.bfloat16)
        ids, w = route(L, M, step)
        yc = store.forward_cached(key, x, ids, w)
        nfill = int(store.fill_list[0].item())
        yh = store.forward(key, x, ids, w, ptrs=host_tab[key])
        if not torch.equal(yc, yh):
            bad += 1
        misses_total += nfill
        if step % 50 == 0:
            print({"step": step, "layer": L, "M": M, "fills": nfill, "equal": torch.equal(yc, yh)}, flush=True)
torch.xpu.synchronize()
# static ops after device-cache activity must not clobber device-owned slots
k0 = f"layers.{layers[0]}.mlp.experts"
store.sync_mirror()
owned = (store.slot_of[k0] >= 0).nonzero().flatten().tolist()
store.evict(k0, owned[:32])
store.make_resident(k0, [e for e in range(E) if store.slot_of[k0][e] < 0][:32])
for step in range(args.steps, args.steps + 20):
    for L in layers:
        key = f"layers.{L}.mlp.experts"
        x = (torch.randn((1, H), device=dev) * 0.5).to(torch.bfloat16)
        ids, w = route(L, 1, step)
        if not torch.equal(store.forward_cached(key, x, ids, w), store.forward(key, x, ids, w, ptrs=host_tab[key])):
            bad += 1
torch.xpu.synchronize()
# consistency
sk, sod, pa = store.slot_key.cpu(), store.slot_of_dev.cpu(), store.ptrs_all.cpu()
incons = 0
for s in range(args.slots):
    k = int(sk[s])
    if k < 0:
        continue
    li, e = divmod(k, E)
    key = f"layers.{layers[li]}.mlp.experts"
    if int(sod[k]) != s or int(pa[k]) != store.slot_ptr(s) or not torch.equal(store.slots[s].cpu(), store.host[key][e]):
        incons += 1
for li, L in enumerate(layers):
    key = f"layers.{L}.mlp.experts"
    for e in range(E):
        k = li * E + e
        if int(sod[k]) < 0 and int(pa[k]) != int(store._host_ptr[key][e]):
            incons += 1
occupied = int((sk >= 0).sum())
summary = {"steps": args.steps, "layers": len(layers), "slots": args.slots, "zipf": args.zipf, "unequal_outputs": bad,
           "inconsistent_entries": incons, "occupied_slots": occupied, "fills_total": misses_total}
print(summary, flush=True)
res["summary"] = summary

if args.bench:
    def bench(fn, iters=300):
        for i in range(20):
            fn(i)
        torch.xpu.synchronize()
        t = time.perf_counter()
        for i in range(iters):
            fn(i)
        torch.xpu.synchronize()
        return (time.perf_counter() - t) / iters * 1e6

    for M in (1, 4):
        xs = (torch.randn((M, H), device=dev) * 0.5).to(torch.bfloat16)
        rts = {L: [route(L, M, 10_000 + i) for i in range(256)] for L in layers}
        # hit rate in steady state (sync per call: measurement only)
        hits = fills = 0
        for i in range(256):
            for L in layers:
                ids, w = rts[L][i]
                store.forward_cached(f"layers.{L}.mlp.experts", xs, ids, w)
                f = int(store.fill_list[0].item())
                u = len(torch.unique(ids))
                fills += f
                hits += u - f
        hit_rate = hits / (hits + fills)

        def cached(i):
            for L in layers:
                ids, w = rts[L][i % 256]
                store.forward_cached(f"layers.{L}.mlp.experts", xs, ids, w)

        def zero_copy(i):
            for L in layers:
                ids, w = rts[L][i % 256]
                store.forward(f"layers.{L}.mlp.experts", xs, ids, w, ptrs=host_tab[f"layers.{L}.mlp.experts"])

        uc = bench(cached) / len(layers)
        uz = bench(zero_copy) / len(layers)
        rec = {"M": M, "slots": args.slots, "slot_frac_of_experts": round(args.slots / (E * len(layers)), 3),
               "zipf": args.zipf, "hit_rate": round(hit_rate, 3), "cached_us_per_layer": round(uc, 1),
               "zero_copy_us_per_layer": round(uz, 1)}
        res["bench"].append(rec)
        print(rec, flush=True)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
sys.exit(1 if bad or incons else 0)
