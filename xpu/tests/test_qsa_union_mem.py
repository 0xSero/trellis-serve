"""QSA union sparse attention (prefill) on XPU: peak memory, time and output of the per-KV-head
mode (EXL3_QSA_UNION_MEM=perhead) against orig.

Flash-Next full-attention shapes: 24 query heads, 2 KV heads, head_dim 256, fp8 (e4m3) K/V cache. One 2048-row
prefill chunk per case; each row selects 2048 slots (the indexer's token budget). Two slot patterns per context:
  random : slots uniform over the context -> union close to the whole context (worst case)
  local  : each row takes a recent window of 512 slots plus 1536 slots from 96 random 16-slot blocks (realistic)
Both modes run warm-up + 3 timed repetitions (peak allocated above the inputs, median ms); outputs are compared with
the orig mode bit for bit (and by max abs difference).
  python test_qsa_union_mem.py [--out FILE]
Exit 0 only if perhead is within 2e-3 max abs difference of orig in every case and has the lower peak from 64K.
"""
import argparse, json, os, statistics, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from exl3xpu.qsa_xpu import qsa_sparse_attention_union

ap = argparse.ArgumentParser()
ap.add_argument("--out")
a = ap.parse_args()
dev = torch.device("xpu", 0)
R, HQ, HK, D, TOPK = 2048, 24, 2, 256, 2048
g = torch.Generator(device="cpu").manual_seed(0)


def slots_random(ctx):
    return torch.randint(0, ctx, (R, TOPK), generator=g)


def slots_local(ctx):
    rows = []
    for i in range(R):
        pos = ctx - R + i
        recent = torch.arange(max(0, pos - 511), pos + 1)
        blocks = torch.randint(0, max(1, (pos - 512) // 16), (96,), generator=g) * 16
        far = (blocks[:, None] + torch.arange(16)[None, :]).reshape(-1)
        s = torch.cat([recent, far])[:TOPK]
        if s.numel() < TOPK:
            s = torch.cat([s, torch.full((TOPK - s.numel(),), -1)])
        rows.append(s)
    return torch.stack(rows)


def run(mode, q, kc, vc, ts):
    qsa_sparse_attention_union(q, kc, vc, ts, mode=mode)               # warm-up
    torch.xpu.synchronize()
    peaks, times = [], []
    for _ in range(3):
        torch.xpu.empty_cache(); torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(dev)
        base = torch.xpu.memory_allocated(dev)
        t = time.perf_counter()
        o = qsa_sparse_attention_union(q, kc, vc, ts, mode=mode)
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
        peaks.append((torch.xpu.max_memory_allocated(dev) - base) / 2**20)
    return o.cpu(), max(peaks), statistics.median(times)


results, ok = [], True
for ctx in (16384, 65536, 131072, 200000):
    kc = torch.randn(ctx, HK, D, generator=g).to(dev).to(torch.float8_e4m3fn)
    vc = torch.randn(ctx, HK, D, generator=g).to(dev).to(torch.float8_e4m3fn)
    q = torch.randn(R, HQ, D, generator=g).to(dev, torch.bfloat16)
    for pattern, fn in (("random", slots_random), ("local", slots_local)):
        ts = fn(ctx).to(dev)
        nu = int(torch.unique(ts[ts >= 0]).numel())
        o_ref, p_ref, t_ref = run("orig", q, kc, vc, ts)
        row = dict(context=ctx, pattern=pattern, union_slots=nu, orig=dict(peak_mib=round(p_ref), median_ms=round(t_ref, 1)))
        for mode in ("perhead",):
            o, pk, tm = run(mode, q, kc, vc, ts)
            same = torch.equal(o, o_ref)
            diff = float((o.float() - o_ref.float()).abs().max())
            row[mode] = dict(peak_mib=round(pk), median_ms=round(tm, 1), bit_identical=same, max_abs_diff=diff)
            ok &= diff <= 2e-3 and (ctx < 65536 or pk < p_ref)
        print(json.dumps(row), flush=True)
        results.append(row)
    del kc, vc, q
    torch.xpu.empty_cache()
if a.out:
    json.dump(dict(results=results, perhead_ok=ok), open(a.out, "w"), indent=1)
sys.exit(0 if ok else 1)
