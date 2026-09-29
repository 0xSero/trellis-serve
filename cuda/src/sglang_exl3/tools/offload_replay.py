"""K05 validation harness for OffloadRuntime / Exl3OffloadMoEMethod: 48-layer MoE-only replay of the K02 routing traces.

    python -m sglang_exl3.tools.offload_replay <model> --mode load|correct|decode|prefill [--slots N] [--traces DIR]

  load     HostExpertStore for all 48 layers: load time breakdown; records of sampled (layer, expert) == om.expert_record
           built from the checkpoint tensors, byte for byte
  correct  runtime built through Exl3OffloadMoEMethod (SGLang quant-method object on a stand-in layer module);
           decode path (cache + zero-copy + admission, several steps) and staged prefill path == stacked reference
           (marlin_moe.run, same align output) bit for bit on real layers; a graph-captured 3-layer decode step replayed
           with new routings == stacked
  decode   the K02 decode stream (22 requests, K03 order) through all 48 layers, one CUDA graph per token step (static
           routing buffers refreshed per step), synthetic hidden states; MoE ms/token + hit rate vs the K03 prediction
  prefill  4k / 8k / 16k-token chunks (routing = long-context prefill traces) through all 48 layers after a decode
           warm-up of the cache; MoE tok/s; plus the decode-vs-staged crossover sweep (32..1024 tokens)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time
import types

import numpy as np
import torch

from ..kernels import marlin_moe, offload_moe as om
from ..kernels.offload_runtime import OffloadRuntime, _align
from ..kernels.offload_store import HostExpertStore
from .moe_parity_sm86 import load_layer

L_ALL = list(range(48))


def eq16(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def build_store(model, layers=L_ALL):
    st = HostExpertStore(model, layers, 512, 2560, 640, 3)
    s = st.load()
    return st, s


def load_traces(d):
    reqs = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        reqs.append({"name": os.path.basename(f)[:-4], "decode_ids": z["decode_ids"], "decode_w": z["decode_w"],
                     "prefill_ids": z["prefill_ids"], "prefill_w": z["prefill_w"]})
    order = np.random.default_rng(0).permutation(len(reqs))       # = K03 simulate_cache order
    return [reqs[i] for i in order]


def mode_load(a):
    t0 = time.time()
    st, s = build_store(a.model)
    out = {"load": {k: v for k, v in vars(s).items() if k != "per_layer_register_s"},
           "register_per_layer_s_median": statistics.median(s.per_layer_register_s),
           "GB": st.L * st.bank_bytes / 1e9, "wall_s": time.time() - t0}
    ok = True
    for l in (0, 23, 47):
        _, t, _ = load_layer(a.model, l, 512, "cpu")
        for e in (0, 255, 511):
            ok &= torch.equal(st.bank(l)[e], om.expert_record(t["gate_proj"], t["up_proj"], t["down_proj"], e, st.lay))
    out["records_exact"] = bool(ok)
    print("LOAD", json.dumps(out), flush=True)
    st.release()
    return out


def fake_method(model, layer_id, slots):
    """Exl3OffloadMoEMethod on a stand-in torch module, exactly as SGLang drives it (create_weights -> loader calls
    -> create_moe_runner -> process_weights_after_loading -> apply)."""
    os.environ["SGLANG_EXL3_MOE_OFFLOAD"] = "1"
    os.environ["SGLANG_EXL3_OFFLOAD_SLOTS"] = str(slots)
    os.environ["SGLANG_EXL3_OFFLOAD_STAGING_PARTS"] = str(getattr(fake_method, "parts", 1))
    os.environ["SGLANG_EXL3_OFFLOAD_PREFILL_SUBCHUNK"] = str(getattr(fake_method, "subchunk", 1 << 30))
    from ..sglang_glue.config import Exl3Config
    from ..sglang_glue.offload_moe_method import Exl3OffloadMoEMethod
    cfg = Exl3Config({}, model)
    if not getattr(cfg, "modules", None):
        cfg._scan(model)
    try:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: F401
    except Exception:
        pass
    prefix = f"model.layers.{layer_id}.mlp.experts"
    from ..sglang_glue.config import _expert_of
    experts = {k: v for k, v in cfg.modules.items() if _expert_of(k, prefix)}
    m = Exl3OffloadMoEMethod(cfg, prefix, experts)
    layer = torch.nn.Module()
    m.create_weights(layer, 512, 2560, 640, torch.bfloat16)
    p = getattr(layer, "w13_trellis")
    p.weight_loader(p, torch.zeros(4), "x", shard_id="w1", expert_id=0)       # discarded
    m.create_moe_runner(layer, types.SimpleNamespace(activation="silu", is_gated=True, apply_router_weight_on_input=False,
                                                     routed_scaling_factor=None, num_fused_shared_experts=0, top_k=10))
    m.process_weights_after_loading(layer)
    return m, layer


def mode_correct(a):
    res = {}
    ms, layers = {}, {}
    fake_method.parts = a.parts
    fake_method.subchunk = a.subchunk
    for l in (3, 17, 40):
        ms[l], layers[l] = fake_method(a.model, l, a.slots)
    rt = layers[3].exl3_offload
    res["discarded_loader_calls"] = int(getattr(layers[3], "exl3_discarded", 0))
    packs = {}
    for l in (3, 17, 40):
        _, t, cb = load_layer(a.model, l, 512, "cpu")
        packs[l] = marlin_moe.prepare(t["gate_proj"], t["up_proj"], t["down_proj"], cb)
    gen = torch.Generator().manual_seed(0)
    ok_dec = ok_pre = True
    for step in range(12):
        T = (1, 1, 4, 16)[step % 4]
        for l in (3, 17, 40):
            ids = torch.stack([torch.randperm(512, generator=gen)[:10] for _ in range(T)]).int().cuda()
            w = torch.softmax(torch.randn((T, 10), generator=gen), -1).cuda()
            x = (torch.randn((T, 2560), generator=gen) * 0.5).bfloat16().cuda()
            block = marlin_moe.moe_block_size(T, 10, 512)
            al = _align(ids, block, 512)
            disp = types.SimpleNamespace(hidden_states=x, topk_output=types.SimpleNamespace(topk_weights=w, topk_ids=ids))
            y = ms[l].apply(layers[l], disp).hidden_states            # through the SGLang method (own align)
            y2 = rt.forward(layers[l].exl3_store_index, x, ids, w, routing=al, force="decode")
            ys = marlin_moe.run(x, w, ids, *al, block, packs[l])
            ok_dec &= eq16(y, ys) and eq16(y2, ys)
    res["decode_eq_stacked"] = bool(ok_dec)
    for T in (512, 2048, 4096):
        for l in (3, 17, 40):
            ids = torch.stack([torch.randperm(512, generator=gen)[:10] for _ in range(T)]).int().cuda()
            w = torch.softmax(torch.randn((T, 10), generator=gen), -1).cuda()
            x = (torch.randn((T, 2560), generator=gen) * 0.5).bfloat16().cuda()
            block = marlin_moe.moe_block_size(T, 10, 512)
            al = _align(ids, block, 512)
            y = rt.forward(layers[l].exl3_store_index, x, ids, w, routing=al, force="prefill")
            ys = marlin_moe.run(x, w, ids, *al, block, packs[l])
            ok_pre &= eq16(y, ys)
            res.setdefault("prefill_rel_diff_max", 0.0)
            res["prefill_rel_diff_max"] = max(res["prefill_rel_diff_max"],
                                              float((y.float() - ys.float()).abs().mean() / ys.float().abs().mean()))
            res["prefill_finite"] = bool(torch.isfinite(y).all()) and res.get("prefill_finite", True)
    res["prefill_eq_stacked"] = bool(ok_pre)
    # graph: 3-layer decode step
    T = 1
    xs = [(torch.randn((T, 2560), generator=gen) * 0.5).bfloat16().cuda() for _ in range(3)]
    idb = [torch.zeros((T, 10), dtype=torch.int32, device="cuda") for _ in range(3)]
    wb = [torch.full((T, 10), 0.1, device="cuda") for _ in range(3)]
    lids = (3, 17, 40)
    for i in range(3):
        idb[i].copy_(torch.randperm(512, generator=gen)[:10].view(1, 10))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i, l in enumerate(lids):
            rt.forward(layers[l].exl3_store_index, xs[i], idb[i], wb[i])
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ys = [rt.forward(layers[l].exl3_store_index, xs[i], idb[i], wb[i]) for i, l in enumerate(lids)]
    ok_g = True
    for rep in range(20):
        for i in range(3):
            idb[i].copy_(torch.randperm(512, generator=gen)[:10].view(1, 10).int())
            wb[i].copy_(torch.softmax(torch.randn((1, 10), generator=gen), -1))
            xs[i].copy_((torch.randn((1, 2560), generator=gen) * 0.5).bfloat16())
        g.replay(); torch.cuda.synchronize()
        for i, l in enumerate(lids):
            block = marlin_moe.moe_block_size(1, 10, 512)
            ok_g &= eq16(ys[i], marlin_moe.run(xs[i], wb[i], idb[i], *_align(idb[i], block, 512), block, packs[l]))
    res["graph_eq_stacked"] = bool(ok_g)
    res["stats"] = {k: v for k, v in rt.stats().items() if k in ("decode_hit_rate", "resident", "slots")}
    res["parts"] = a.parts
    res["subchunk"] = a.subchunk
    res["pass"] = ok_dec and ok_g and (ok_pre if (a.parts == 1 and a.subchunk >= 4096) else (res["prefill_rel_diff_max"] < 2e-3 and res["prefill_finite"]))
    print("CORRECT", json.dumps(res), flush=True)
    return res


def mode_decode(a):
    st, s = build_store(a.model)
    print("store loaded", round(s.seconds_total, 1), "s", flush=True)
    reqs = load_traces(a.traces)
    ids_all = np.concatenate([r["decode_ids"] for r in reqs]).astype(np.int32)           # [N, 48, 10]
    w_all = np.concatenate([r["decode_w"] for r in reqs]).astype(np.float32)
    if a.steps:
        ids_all, w_all = ids_all[: a.steps], w_all[: a.steps]
    N = ids_all.shape[0]
    ids_d, w_d = torch.from_numpy(ids_all).cuda(), torch.from_numpy(w_all).cuda()
    out = {"steps": N, "results": []}
    for slots in [int(v) for v in a.slots_list.split(",")]:
        rt = OffloadRuntime(st, slots, 2, prefill_min_tokens=10**9, staging=False)
        gen = torch.Generator().manual_seed(0)
        xs = (torch.randn((48, 1, 2560), generator=gen) * 0.5).bfloat16().cuda()
        idb = torch.zeros((48, 1, 10), dtype=torch.int32, device="cuda")
        wb = torch.zeros((48, 1, 10), dtype=torch.float32, device="cuda")
        idb.copy_(ids_d[0].view(48, 1, 10)); wb.copy_(w_d[0].view(48, 1, 10))

        def step():
            for l in range(48):
                rt.forward(l, xs[l], idb[l], wb[l])

        sd = torch.cuda.Stream(); sd.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sd):
            step()
        torch.cuda.current_stream().wait_stream(sd)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        rt.stats(reset=True)
        # restart the cache from empty for a like-for-like comparison with K03 (warm-up forwards admitted a few experts)
        rt.cache.slot_of.fill_(-1); rt.cache.owner.fill_(-1); rt.cache.stamp.zero_(); rt.cache.ref.zero_()
        rt.cache.hand.zero_(); rt.cache.clock.zero_(); rt.cache.admit.zero_()
        e = torch.arange(512, dtype=torch.int64, device="cuda")
        for l in range(48):
            rt.cache.tables[l].copy_(rt.cache.host_bases[l] + (e * st.lay.record_bytes).unsqueeze(1) + rt.cache.offs.unsqueeze(0))
        blk = 2000
        evs = []
        torch.cuda.synchronize()
        t0 = time.time()
        for t in range(N):
            if t % blk == 0:
                ev = torch.cuda.Event(enable_timing=True); ev.record(); evs.append((t, ev))
            idb.copy_(ids_d[t].view(48, 1, 10)); wb.copy_(w_d[t].view(48, 1, 10))
            g.replay()
        ev = torch.cuda.Event(enable_timing=True); ev.record(); evs.append((N, ev))
        torch.cuda.synchronize()
        wall = time.time() - t0
        per_block = [(evs[i][1].elapsed_time(evs[i + 1][1])) / (evs[i + 1][0] - evs[i][0]) for i in range(len(evs) - 1)]
        total_ms = evs[0][1].elapsed_time(evs[-1][1])
        stt = rt.stats()
        h = stt["decode_hit_rate"]
        pred = 48 * (5.5 * 10 * h + 73 * 10 * (1 - h)) / 1000
        r = {"slots": slots, "ms_per_token": total_ms / N, "median_block_ms_per_token": statistics.median(per_block),
             "hit_rate": h, "predicted_ms_from_hit_rate": pred, "wall_s": wall,
             "per_layer_hit_rate": [round(hh / max(1, hh + mm), 3) for hh, mm in zip(stt["decode_hits"], stt["decode_misses"])]}
        out["results"].append(r)
        print("DECODE", json.dumps({k: v for k, v in r.items() if k != "per_layer_hit_rate"}), flush=True)
        del g, rt
        torch.cuda.empty_cache()
    st.release()
    return out


def mode_prefill(a):
    st, s = build_store(a.model)
    reqs = load_traces(a.traces)
    long = [r for r in reqs if r["name"].startswith("L")]
    pre = np.concatenate([r["prefill_ids"] for r in long]).astype(np.int32)                 # [P, 48, 10]
    prw = np.concatenate([r["prefill_w"] for r in long]).astype(np.float32)
    pre[:, 47], prw[:, 47] = pre[:, 46], prw[:, 46]                                        # layer 47 not routed in prefill
    dec = np.concatenate([r["decode_ids"] for r in reqs]).astype(np.int32)[: a.warm_steps]
    decw = np.concatenate([r["decode_w"] for r in reqs]).astype(np.float32)[: a.warm_steps]
    out = {"results": [], "crossover": []}
    rt = OffloadRuntime(st, a.slots, 2, prefill_min_tokens=a.prefill_min, staging_parts=a.parts, prefill_subchunk=a.subchunk)
    out["subchunk"] = a.subchunk
    out["parts"] = a.parts
    out["staging_GB"] = sum(t.numel() for t in rt.staging) / 1e9
    # warm the cache with decode steps (eager is fine here)
    for t in range(dec.shape[0]):
        for l in range(48):
            rt.forward(l, torch.zeros((1, 2560), dtype=torch.bfloat16, device="cuda"),
                       torch.from_numpy(dec[t, l:l + 1]).cuda(), torch.from_numpy(decw[t, l:l + 1]).cuda())
    torch.cuda.synchronize()
    print("warm", rt.stats()["resident"], "resident, hit rate", round(rt.stats()["decode_hit_rate"], 3), flush=True)
    gen = torch.Generator().manual_seed(0)

    def run48(T, off, force):
        ids = torch.from_numpy(pre[off:off + T]).cuda()
        w = torch.from_numpy(prw[off:off + T]).cuda()
        x = (torch.randn((T, 2560), generator=gen) * 0.5).bfloat16().cuda()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for l in range(48):
            rt.forward(l, x, ids[:, l].contiguous(), w[:, l].contiguous(), force=force)
        e1.record(); torch.cuda.synchronize()
        run48.peak_GB = max(getattr(run48, "peak_GB", 0.0), (torch.cuda.max_memory_allocated() - base) / 1e9)
        return e0.elapsed_time(e1)

    for T in [int(v) for v in a.chunks.split(",")]:
        rt.stats(reset=True)
        run48.peak_GB = 0.0
        ts = [run48(T, (i * 3000) % max(1, pre.shape[0] - T), "prefill") for i in range(a.reps + 1)][1:]
        stt = rt.stats()
        copied = np.array(stt["prefill_copied"]).sum() / max(1, np.array(stt["prefill_calls"]).sum())
        r = {"tokens": T, "slots": a.slots, "ms_per_chunk": statistics.median(ts), "tok_per_s": T / statistics.median(ts) * 1e3,
             "copied_experts_per_layer": copied, "ms_all": [round(v, 1) for v in ts],
             "activation_peak_GB": round(run48.peak_GB, 3)}
        out["results"].append(r)
        print("PREFILL", json.dumps(r), flush=True)
    for T in [int(v) for v in a.crossover.split(",") if v]:
        row = {"tokens": T}
        for force in ("decode", "prefill"):
            row[force + "_ms"] = statistics.median([run48(T, 1000 + 97 * i, force) for i in range(3)])
        print("CROSSOVER", json.dumps(row), flush=True)
        out["crossover"].append(row)
    st.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--slots", type=int, default=512)
    ap.add_argument("--slots-list", default="6000,8000,9400")
    ap.add_argument("--traces", default="/w/runs/2026-09-29-K02-traces")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--warm-steps", type=int, default=3000)
    ap.add_argument("--chunks", default="4096,8192,16384")
    ap.add_argument("--crossover", default="32,64,128,256,512,1024")
    ap.add_argument("--prefill-min", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--subchunk", type=int, default=1 << 30)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fn = {"load": mode_load, "correct": mode_correct, "decode": mode_decode, "prefill": mode_prefill}[a.mode]
    with torch.inference_mode(False):
        out = fn(a)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
