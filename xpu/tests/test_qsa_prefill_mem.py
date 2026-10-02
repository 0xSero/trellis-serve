"""QSA indexer prefill on XPU: peak memory, time and outputs of the upstream row chunking vs qsa_prefill_mem.

Runs SGLang's own qsa_mqa_prefill (torch fallback on XPU) and the trellis XPU qsa_fast_topk the way
QSAIndexer.select_prefill_tokens loops, once with upstream chunk sizes and once with the full-peak budget.
Cases (Flash-Next indexer: 4 heads, head_dim 128, compress 4, block_topk 512):
  * single sequence, last 2048 rows of a 16K / 64K / 128K / 262K context
  * packed batch: two sequences in one prefill (nonzero row_starts for the second), rows not a multiple of the chunk
  * early rows: a chunk at the start of a sequence (short and empty rows)
Pass 1 (memory/time): warm-up, then 3 timed repetitions each; outputs stay on the device and are dropped the way the
caller drops them (logits freed after top-k, indices kept); reports peak allocated MiB above the inputs and median ms.
Pass 2 (outputs): logits compared bit for bit and by max abs difference; indices compared both raw and sorted per row
(top-k order may differ; the expander sorts valid entries).
  python test_qsa_prefill_mem.py [--out FILE]
Exit 0 only if, for every case, sorted indices are identical AND logits are bit-identical; otherwise 1 (with the
differences reported), 3 on setup error.
"""
import argparse, json, os, statistics, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--out")
a = ap.parse_args()

from sglang.srt.layers.attention.qsa import qsa_indexer as qi
from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_prefill
from exl3xpu import qsa_xpu
from exl3xpu import qsa_prefill_mem
upstream = qi._qsa_prefill_row_chunk_size
qsa_xpu.install()
topk_fn = qi.qsa_fast_topk
if topk_fn.__module__ != qsa_xpu.__name__:
    print(f"setup error: qsa_fast_topk is {topk_fn.__module__}.{topk_fn.__name__}, not the trellis XPU override")
    sys.exit(3)
dev = torch.device("xpu", 0)
H, D, RATIO, TOPK = 4, 128, 4, 512
g = torch.Generator(device="cpu").manual_seed(0)


def case_single(ctx, rows=2048):
    keys = ctx // RATIO
    pos = torch.arange(ctx - rows, ctx)
    return keys, torch.zeros(rows, dtype=torch.int32), ((pos + 1) // RATIO).to(torch.int32)


def case_packed():
    # seq A: last 1000 rows of a 40K context (keys [0, 10000)); seq B: last 1300 rows of a 30K context whose keys are
    # packed after A's (keys [10000, 17500)); 2300 rows total, not a multiple of any chunk size
    ka, kb = 40000 // RATIO, 30000 // RATIO
    pa, pb = torch.arange(40000 - 1000, 40000), torch.arange(30000 - 1300, 30000)
    rs = torch.cat([torch.zeros(1000), torch.full((1300,), ka)]).to(torch.int32)
    re = torch.cat([(pa + 1) // RATIO, ka + (pb + 1) // RATIO]).to(torch.int32)
    return ka + kb, rs, re


def case_early():
    # first 2048 rows of a sequence: rows 0-2 see no complete block (empty), later rows see few keys
    pos = torch.arange(0, 2048)
    return 2048 // RATIO, torch.zeros(2048, dtype=torch.int32), ((pos + 1) // RATIO).to(torch.int32)


def loop(chunker, q, k, rs, re, keep):
    rows = q.shape[0]
    size = chunker(rows, k.shape[0], q.shape[1])
    lg_all, idx_all = [], []
    for s in range(0, rows, size):
        e = min(s + size, rows)
        lg = qsa_mqa_prefill(q[s:e], k, rs[s:e], re[s:e])
        idx = topk_fn(lg, rs[s:e], re[s:e], topk=TOPK)
        idx_all.append(idx)
        if keep:
            lg_all.append(lg)
        del lg
    return size, lg_all, idx_all


def measure(chunker, q, k, rs, re):
    loop(chunker, q, k, rs, re, False)                                 # warm-up
    torch.xpu.synchronize()
    peaks, times = [], []
    for _ in range(3):
        torch.xpu.empty_cache(); torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(dev)
        base = torch.xpu.memory_allocated(dev)
        t = time.perf_counter()
        size, _, idx = loop(chunker, q, k, rs, re, False)
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
        peaks.append((torch.xpu.max_memory_allocated(dev) - base) / 2**20)
        del idx
    return size, max(peaks), statistics.median(times)


cases = [(f"single_{c // 1024}k", *case_single(c)) for c in (16384, 65536, 131072, 262144)]
cases += [("packed_2seq", *case_packed()), ("early_rows", *case_early())]
results, ok = [], True
for name, keys, rs, re in cases:
    rows = rs.numel()
    q = torch.randn(rows, H, D, generator=g).to(dev, torch.bfloat16)
    k = torch.randn(keys, 1, D, generator=g).to(dev, torch.bfloat16)
    rs_d, re_d = rs.to(dev), re.to(dev)
    mu = measure(upstream, q, k, rs_d, re_d)
    mp = measure(qsa_prefill_mem.row_chunk_size, q, k, rs_d, re_d)
    _, lu, iu = loop(upstream, q, k, rs_d, re_d, True)
    _, lp, ip = loop(qsa_prefill_mem.row_chunk_size, q, k, rs_d, re_d, True)
    lu, lp = torch.cat(lu).cpu(), torch.cat(lp).cpu()
    iu, ip = torch.cat(iu).cpu(), torch.cat(ip).cpu()
    finite = torch.isfinite(lu) & torch.isfinite(lp)
    same_inf = torch.equal(torch.isfinite(lu), torch.isfinite(lp))
    maxdiff = float((lu[finite] - lp[finite]).abs().max()) if finite.any() else 0.0
    logits_bit = torch.equal(lu, lp)
    idx_raw = torch.equal(iu, ip)
    idx_sorted = torch.equal(iu.sort(dim=-1).values, ip.sort(dim=-1).values)
    ok &= idx_sorted and logits_bit
    row = dict(case=name, rows=rows, keys=keys,
               upstream=dict(rows_per_chunk=mu[0], peak_mib=round(mu[1]), median_ms=round(mu[2], 1)),
               patched=dict(rows_per_chunk=mp[0], peak_mib=round(mp[1]), median_ms=round(mp[2], 1)),
               logits_bit_identical=logits_bit, logits_max_abs_diff=maxdiff, inf_pattern_identical=same_inf,
               indices_raw_identical=idx_raw, indices_sorted_identical=idx_sorted)
    print(json.dumps(row), flush=True)
    results.append(row)
    del q, k, lu, lp, iu, ip
    torch.xpu.empty_cache()
if a.out:
    json.dump(dict(results=results, passed=ok), open(a.out, "w"), indent=1)
sys.exit(0 if ok else 1)
