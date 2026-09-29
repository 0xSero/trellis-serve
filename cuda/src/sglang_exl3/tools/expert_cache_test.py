"""ExpertCache (device-side slot cache + fused admission) on REAL layers: correctness under heavy eviction, CUDA-graph
replay, and per-layer timing vs hit rate.

    python -m sglang_exl3.tools.expert_cache_test /models/<model> --layers 0,1,2,3 --slots 24 --steps 40

Checks after every step: every layer output == stacked grouped path (bit for bit); cache invariants: owner/slot_of
consistent, table row = slot address iff resident else host bank row, admit tables all zero after commit, every
resident slot's bytes == its host record. Graph: one captured multi-layer step replayed with new routings.
"""
from __future__ import annotations

import argparse
import statistics

import torch

from ..kernels import marlin_moe, offload_moe as om
from .offload_moe_bench import Layer, align, eq16


def zipf_routing(tokens, top_k, e, gen, alpha=1.1, perm=None):
    p = 1.0 / torch.arange(1, e + 1, dtype=torch.float64) ** alpha
    if perm is not None:
        p = p[perm]
    ids = torch.multinomial(p.expand(tokens, -1), top_k, replacement=False, generator=gen).to(torch.int32)
    w = torch.softmax(torch.randn((tokens, top_k), generator=gen), dim=-1)
    return ids.cuda(), w.cuda()


def invariants(cache: om.ExpertCache, layers) -> dict:
    L, E, S, rec = cache.L, cache.E, cache.S, cache.lay.record_bytes
    slot_of, owner = cache.slot_of.cpu(), cache.owner.cpu()
    tables, offs = cache.tables.cpu(), cache.offs.cpu()
    arena_base = cache.arena.data_ptr()
    ok_owner = all(int(owner[int(slot_of[g])]) == g for g in range(L * E) if slot_of[g] >= 0) and \
        all(int(slot_of[int(owner[s])]) == s for s in range(S) if owner[s] >= 0)
    hb = cache.host_bases.cpu()
    exp = torch.where(slot_of.view(L, E, 1) >= 0, arena_base + slot_of.view(L, E, 1).long() * rec,
                      hb.view(L, 1, 1) + torch.arange(E).view(1, E, 1) * rec) + offs.view(1, 1, -1)
    ok_tables = torch.equal(tables, exp)
    ok_admit = bool((cache.admit == 0).all())
    arena = cache.arena.cpu()
    ok_bytes = all(torch.equal(arena[s], layers[int(owner[s]) // E].host[int(owner[s]) % E]) for s in range(S) if owner[s] >= 0)
    return {"owner": ok_owner, "tables": bool(ok_tables), "admit_clear": ok_admit, "slot_bytes": bool(ok_bytes),
            "resident": int((owner >= 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--timing-slots", default="0,64,256,1024,2048")
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(0)
    lids = [int(v) for v in a.layers.split(",")]
    layers = [Layer(a.model_dir, l) for l in lids]
    E, lay, cb = layers[0].e, layers[0].lay, layers[0].cb
    perms = [torch.randperm(E, generator=gen) for _ in lids]
    cache = om.ExpertCache(len(lids), E, lay, [Ly.host_base for Ly in layers], a.slots)

    def layer_step(i, x, ids, w):
        block = marlin_moe.moe_block_size(x.shape[0], 10, E)
        return cache.run(i, x, w, ids, *align(ids, block, E), block, cb)

    # (1) eager steps with heavy eviction
    ok = True
    for step in range(a.steps):
        T = (1, 1, 4, 16)[step % 4]
        for i, Ly in enumerate(layers):
            ids, w = zipf_routing(T, 10, E, gen, perm=perms[i])
            x = (torch.randn((T, lay.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
            y = layer_step(i, x, ids, w)
            ok &= eq16(y, Ly.run_stack(x, ids, w))
        if step % 5 == 4 or step == a.steps - 1:
            inv = invariants(cache, layers)
            ok &= all(v for k, v in inv.items() if k != "resident")
            print("step", step, "outputs_eq_stacked", bool(ok), inv, "hit_rate", [round(v, 3) for v in cache.hit_rate().tolist()], flush=True)
    print("EAGER", "PASS" if ok else "FAIL", flush=True)

    # (2) CUDA graph: all layers of one decode step captured once, replayed with new routings
    T = 1
    xs = [(torch.randn((T, lay.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda() for _ in lids]
    rs = [zipf_routing(T, 10, E, gen, perm=perms[i]) for i in range(len(lids))]
    ids_s, w_s = [r[0].clone() for r in rs], [r[1].clone() for r in rs]
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(len(lids)):
            layer_step(i, xs[i], ids_s[i], w_s[i])
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ys = [layer_step(i, xs[i], ids_s[i], w_s[i]) for i in range(len(lids))]
    gok = True
    for rep in range(30):
        for i in range(len(lids)):
            r_ids, r_w = zipf_routing(T, 10, E, gen, perm=perms[i])
            ids_s[i].copy_(r_ids); w_s[i].copy_(r_w)
            xs[i].copy_((torch.randn((T, lay.hidden), generator=gen) * 0.5).to(torch.bfloat16))
        g.replay(); torch.cuda.synchronize()
        for i, Ly in enumerate(layers):
            gok &= eq16(ys[i], Ly.run_stack(xs[i], ids_s[i], w_s[i]))
    inv = invariants(cache, layers)
    gok &= all(v for k, v in inv.items() if k != "resident")
    print("GRAPH", "PASS" if gok else "FAIL", inv, flush=True)

    # (3) timing: graph-replayed 1-token step over the layers, steady state, per slot budget (zipf routing)
    for slots in [int(s) for s in a.timing_slots.split(",")]:
        c2 = om.ExpertCache(len(lids), E, lay, [Ly.host_base for Ly in layers], slots)
        cache_ref = cache
        cache = c2
        with torch.cuda.stream(side):
            for i in range(len(lids)):
                layer_step(i, xs[i], ids_s[i], w_s[i])
        torch.cuda.current_stream().wait_stream(side)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            for i in range(len(lids)):
                layer_step(i, xs[i], ids_s[i], w_s[i])
        routs = [[zipf_routing(1, 10, E, gen, perm=perms[i]) for i in range(len(lids))] for _ in range(260)]
        ts = []
        for k, rr in enumerate(routs):
            for i in range(len(lids)):
                ids_s[i].copy_(rr[i][0]); w_s[i].copy_(rr[i][1])
            if k == 60:
                c2.stats.zero_()          # warm-up done: count hits from here
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g2.replay(); e1.record(); torch.cuda.synchronize()
            if k >= 60:
                ts.append(e0.elapsed_time(e1) * 1000 / len(lids))
        hr = c2.hit_rate().mean().item()
        print(f"timing slots={slots} ({slots / len(lids):.0f}/layer): {statistics.median(ts):.1f} us per layer, "
              f"hit rate {hr:.3f} (zipf 1.1 routing, 1 token)", flush=True)
        cache = cache_ref
        del c2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
