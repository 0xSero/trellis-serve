"""QSA union attention, EXL3_QSA_UNION_MEM=auto vs orig at a 128 MiB score budget
(EXL3_QSA_DENSE_BYTES, as in the long-context serving measurements).

auto runs the per-KV-head union when the union has >= EXL3_QSA_UNION_PERHEAD_MIN (32768) slots and the original path
otherwise. Cases (2048 rows x 2048 selected slots, 24 q heads, 2 KV heads, head_dim 256, fp8 cache): local slot
pattern at 8K, 16K, 24K contexts (union below the threshold: must be bit-identical) and 64K, 128K, 200K (per-head:
numerical tolerance), plus the random pattern at 200K (worst-case union). Reports peak above inputs, median ms, and
for the per-head cases max abs diff, mean abs diff and relative RMS error vs orig.
  python test_qsa_union_auto.py [--out FILE]
Exit 0 only if below-threshold cases are bit-identical and per-head cases have max abs diff <= 2e-3 and relative RMS
error <= 1e-3.
"""
import argparse, json, os, statistics, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from exl3xpu import qsa_xpu as qx

ap = argparse.ArgumentParser()
ap.add_argument("--out")
a = ap.parse_args()
qx._DENSE_SCORE_BYTES = 128 << 20                                       # EXL3_QSA_DENSE_BYTES of the serving measurements
dev = torch.device("xpu", 0)
R, HQ, HK, D, TOPK = 2048, 24, 2, 256, 2048
g = torch.Generator(device="cpu").manual_seed(0)


def slots_local(ctx):
    rows = []
    for i in range(R):
        pos = ctx - R + i
        recent = torch.arange(max(0, pos - 511), pos + 1)
        blocks = torch.randint(0, max(1, (pos - 512) // 16), (96,), generator=g) * 16
        far = (blocks[:, None] + torch.arange(16)[None, :]).reshape(-1)
        rows.append(torch.cat([recent, far])[:TOPK])
    return torch.stack(rows)


def run(mode, q, kc, vc, ts):
    qx.qsa_sparse_attention_union(q, kc, vc, ts, mode=mode)
    torch.xpu.synchronize()
    peaks, times = [], []
    for _ in range(3):
        torch.xpu.empty_cache(); torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(dev)
        base = torch.xpu.memory_allocated(dev)
        t = time.perf_counter()
        o = qx.qsa_sparse_attention_union(q, kc, vc, ts, mode=mode)
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
        peaks.append((torch.xpu.max_memory_allocated(dev) - base) / 2**20)
    return o.cpu().float(), max(peaks), statistics.median(times)


cases = [(c, "local") for c in (8192, 16384, 24576, 65536, 131072, 200000)] + [(200000, "random")]
results, ok = [], True
for ctx, pattern in cases:
    kc = torch.randn(ctx, HK, D, generator=g).to(dev).to(torch.float8_e4m3fn)
    vc = torch.randn(ctx, HK, D, generator=g).to(dev).to(torch.float8_e4m3fn)
    q = torch.randn(R, HQ, D, generator=g).to(dev, torch.bfloat16)
    ts = (slots_local(ctx) if pattern == "local" else torch.randint(0, ctx, (R, TOPK), generator=g)).to(dev)
    nu = int(torch.unique(ts[ts >= 0]).numel())
    o_ref, p_ref, t_ref = run("orig", q, kc, vc, ts)
    o, pk, tm = run("auto", q, kc, vc, ts)
    perhead = nu >= qx._UNION_PERHEAD_MIN
    d = (o - o_ref).abs()
    rel_rms = float(((o - o_ref).pow(2).mean() / o_ref.pow(2).mean().clamp_min(1e-30)).sqrt())
    row = dict(context=ctx, pattern=pattern, union_slots=nu, auto_path="perhead" if perhead else "orig",
               orig=dict(peak_mib=round(p_ref), median_ms=round(t_ref, 1)), auto=dict(peak_mib=round(pk), median_ms=round(tm, 1)),
               bit_identical=torch.equal(o, o_ref), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
               rel_rms_err=rel_rms)
    row["pass"] = row["bit_identical"] if not perhead else (row["max_abs_diff"] <= 2e-3 and rel_rms <= 1e-3)
    ok &= row["pass"]
    print(json.dumps(row), flush=True)
    results.append(row)
    del kc, vc, q, ts
    torch.xpu.empty_cache()
if a.out:
    json.dump(dict(results=results, passed=ok), open(a.out, "w"), indent=1)
sys.exit(0 if ok else 1)
