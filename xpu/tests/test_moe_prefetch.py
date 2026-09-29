"""
X006: next-layer expert prefetch on a side queue, overlapped with the (simulated) non-MoE work of the layer.

Token loop over L layers, M=1: compute stream = [other work (a bf16 GEMM sized to ~--other-us)] -> [wait prefetch(l)]
-> forward_cached(l); side stream = [wait MoE(l-1) done] -> prefetch(l, predicted ids). Predictor accuracy p: each of
the 10 predicted ids is the true one with probability p, else a random expert. Outputs checked bitwise against the
zero-copy result; slot/table consistency checked at the end.

  python3 tests/test_moe_prefetch.py --layers 8 --slots 1024 --tokens 60
"""
import os, sys, json, time, argparse
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from safetensors import safe_open
from exl3xpu.moe_offload import ExpertStore, pack_expert

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "/model"))
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--slots", type=int, default=1024)
ap.add_argument("--tokens", type=int, default=60)
ap.add_argument("--other-us", type=float, default=200.0)
ap.add_argument("--zipf", type=float, default=0.5)
ap.add_argument("--out", default="")
a = ap.parse_args()
H, I, K, E, TOPK = 2560, 640, 3, 512, 10
dev = torch.device("xpu", 0)
idx = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
handles = {}


def get(n):
    fn = idx[n]
    if fn not in handles:
        handles[fn] = safe_open(f"{a.model}/{fn}", "pt", device="cpu")
    return handles[fn].get_tensor(n)


def build():
    st = ExpertStore(H, I, K, E, a.slots, dev, max_layers=a.layers)
    for L in range(a.layers):
        h = st.add_layer(L)
        for e in range(E):
            p = f"model.language_model.layers.{L}.mlp.experts.{e}."
            t = {pr: {s: get(p + pr + "." + s) for s in ("trellis", "suh", "svh")} for pr in ("gate_proj", "up_proj", "down_proj")}
            pack_expert(t["gate_proj"], t["up_proj"], t["down_proj"], K, out=h[e])
    return st


store = build()
host_tab = {L: store._host_ptr[L].to(dev) for L in range(a.layers)}
pop = {L: -a.zipf * torch.log1p(torch.randperm(E, generator=torch.Generator().manual_seed(100 + L)).float()) for L in range(a.layers)}
# other work sized to ~other_us
A = torch.randn((1024, 1024), device=dev, dtype=torch.bfloat16)
Bm = torch.randn((1024, 1024), device=dev, dtype=torch.bfloat16)
# calibrate by DEVICE time (events over a queue of GEMMs), not host wall time (Python launch cost dominates)
for _ in range(20):
    A @ Bm
torch.xpu.synchronize()
e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
e0.record()
for _ in range(200):
    A @ Bm
e1.record()
torch.xpu.synchronize()
one = e0.elapsed_time(e1) / 200 * 1e3
reps = max(1, round(a.other_us / one))
print(f"other work: {reps} x 1024^2 GEMM = ~{reps * one:.0f} us device time per layer", flush=True)


def other():
    for _ in range(reps):
        A @ Bm


def routes(tok):
    out = []
    for L in range(a.layers):
        g = torch.Generator().manual_seed(tok * 1000 + L)
        w, ids = torch.topk(torch.softmax(torch.randn((1, E), generator=g) + pop[L], -1), TOPK, -1)
        out.append((ids.to(torch.int32), (w / w.sum(-1, keepdim=True)).float()))
    return out


def predict(ids, p, tok, L):
    g = torch.Generator().manual_seed(tok * 7919 + L)
    keep = torch.rand(ids.shape, generator=g) < p
    return torch.where(keep, ids, torch.randint(0, E, ids.shape, generator=g, dtype=torch.int32))


res = []
modes = [("none", None), ("copy_engine", 0.0), ("copy_engine", 0.7), ("copy_engine", 1.0)] + \
        ([("kernel", 0.7), ("kernel", 1.0)] if os.environ.get("PF_KERNEL") else [])
for mode, p in modes:
    store.reset_cache()
    if mode == "copy_engine":
        store.start_copy_prefetch()
    cs = torch.xpu.current_stream()
    side = torch.xpu.Stream()
    x = (torch.randn((1, H), device=dev) * 0.5).to(torch.bfloat16)
    bad = 0
    all_routes = [routes(tok) for tok in range(a.tokens)]
    dev_routes = [[(i.to(dev), w.to(dev)) for i, w in r] for r in all_routes]
    preds = [[predict(all_routes[tok][L][0], p, tok, L).to(dev) if p is not None else None for L in range(a.layers)]
             for tok in range(a.tokens)]
    warm = a.tokens // 3
    t0 = None
    for tok in range(a.tokens):
        if tok == warm:
            torch.xpu.synchronize()
            t0 = time.perf_counter()
        for L in range(a.layers):
            other()
            if mode == "kernel":
                cs.wait_stream(side)                      # prefetch of this layer complete before its MoE call
            ids, w = dev_routes[tok][L]
            y = store.forward_cached(L, x, ids, w)
            nt, nl = (tok, L + 1) if L + 1 < a.layers else (tok + 1, 0)
            if mode == "kernel" and nt < a.tokens:
                side.wait_stream(cs)                      # after this MoE call
                with torch.xpu.stream(side):
                    store.prefetch(nl, preds[nt][nl])
            if mode == "copy_engine" and nt < a.tokens:
                store.plan_prefetch(nl, preds[nt][nl])    # compute stream; copies run on the copy engine
            if tok % 10 == 0:
                ref = store.forward(L, x, ids, w, ptrs=host_tab[L])
                bad += int(not torch.equal(y, ref))
    torch.xpu.synchronize()
    dt_ = (time.perf_counter() - t0) / ((a.tokens - warm) * a.layers) * 1e6
    stats = store.stop_copy_prefetch() if mode == "copy_engine" else None
    torch.xpu.synchronize()
    # consistency
    sk, sod, pa = store.slot_key.cpu(), store.slot_of_dev.cpu(), store.ptrs_all.cpu()
    inc = 0
    cats = {"slot_of": 0, "ptr_host": 0, "ptr_other": 0, "content": 0, "pinned": 0}
    sl_ = store.slot_last.cpu()
    for s_ in range(a.slots):
        k = int(sk[s_])
        if k < 0:
            continue
        bad_ = False
        if int(sod[k]) != s_:
            cats["slot_of"] += 1; bad_ = True
        if int(pa[k]) != store.slot_ptr(s_):
            cats["ptr_host" if int(pa[k]) == int(store._host_ptr[k // E][k % E]) else "ptr_other"] += 1; bad_ = True
        if not torch.equal(store.slots[s_].cpu(), store.host[k // E][k % E]):
            cats["content"] += 1; bad_ = True
        if int(sl_[s_]) == 0x7FFFFFFF:
            cats["pinned"] += 1
        inc += bad_
    print("inconsistency categories:", cats, flush=True)
    rec = {"mode": mode, "prefetch_accuracy": p, "worker_plans_copies": stats, "layers": a.layers, "slots": a.slots, "zipf": a.zipf, "other_us": round(reps * one),
           "us_per_layer_incl_other": round(dt_, 1), "unequal_outputs": bad, "inconsistent_slots": inc}
    res.append(rec)
    print(rec, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
sys.exit(1 if any(r["unequal_outputs"] or r["inconsistent_slots"] for r in res) else 0)
