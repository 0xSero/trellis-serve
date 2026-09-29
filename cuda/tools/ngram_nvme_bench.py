"""Host-side benchmark + correctness for the n-gram NVMe tier (sglang_exl3.offload.ngram_nvme RowStore).

Token streams: the 22 K02 workloads (prompts from kernels/cuda_sm86/traces/workloads.json, completions from the K02
npz `text`), tokenised with the model tokenizer. Each request = prefill (chunks of --chunk tokens) then decode (one
token per lookup, 16 rows). Requests are replayed back to back through one store (cache shared across requests).

  python3 tools/ngram_nvme_bench.py --model /models/... --out /w/runs/... [--budgets 0.03,0.06,...] [--io aio]
Modes: --mode hashcheck | sweep | correct | qd
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

EOS = 248044


def build_streams(model, workloads, k02):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
    W = json.load(open(workloads))
    readme = open(os.path.join(model, "README.md")).read()
    out = []
    for cat in ("prose", "code", "long"):
        for t in W[cat]:
            doc = readme if t.get("doc_file") == "MODEL_README" else t.get("doc", "")
            q = (doc + "\n\n" + t["question"]) if doc else t["question"]
            text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            if not t["think"]:
                text += "<think>\n\n</think>\n\n"
            p = tok.encode(text, add_special_tokens=False).ids
            z = np.load(os.path.join(k02, t["name"] + ".npz"))
            c = tok.encode(str(z["text"]), add_special_tokens=False).ids
            out.append(dict(name=t["name"], cat=cat, prompt=p, completion=c))
    return out


def ref_hash(tokens, mult, sizes, offs, hpn, eos=EOS):
    """Port of SGLang Qwen4ExpNGramEmbedding._hash_contexts on unfolded 3-token windows (history eos)."""
    t = torch.tensor([eos, eos] + list(tokens), dtype=torch.long)
    ctx = t.unfold(0, 3, 1)  # [L, 3]
    B, S = ctx.shape
    idx = torch.arange(S)
    shifted = [ctx]
    for n in (1, 2):
        eos_pos = torch.where(ctx == eos, idx, -1)
        prev_incl = torch.cummax(eos_pos, dim=1).values
        prev = torch.cat([eos_pos.new_full((B, 1), -1), prev_incl[:, :-1]], dim=1)
        pos_in_seg = idx.unsqueeze(0) - (prev + 1)
        src = idx - n
        g = torch.clamp(src, min=0).unsqueeze(0).expand(B, -1)
        sh = ctx.gather(1, g)
        valid = (pos_in_seg >= n) & (src.unsqueeze(0) >= 0)
        shifted.append(torch.where(valid, sh, torch.full((), eos)))
    blocks = []
    for ng in (2, 3):
        i0 = (ng - 2) * hpn
        mix = shifted[0] * mult[0]
        for pos in range(1, ng):
            mix = torch.bitwise_xor(mix, shifted[pos] * mult[pos])
        ids = torch.remainder(mix[:, -1:].unsqueeze(-1), sizes[i0:i0 + hpn].view(1, 1, -1)) + offs[i0:i0 + hpn].view(1, 1, -1)
        blocks.append(ids[:, 0])
    return torch.cat(blocks, -1)


def pct(a, q):
    return float(np.percentile(np.asarray(a), q)) if len(a) else float("nan")


def make_table(model, gb, io, max_tokens):
    from sglang_exl3.offload.ngram_nvme import Exl3NgramNvmeTable
    return Exl3NgramNvmeTable(model, ram_gb=gb, io=io, device=None, max_tokens=max_tokens, start_service=False)


def replay(tab, streams, chunk, check=None):
    """Replay all streams; returns per-phase timing + hit stats. check: np.memmap rows for byte verification."""
    st = tab.store
    pre_ms, dec_us, pre_hits, dec_hits = [], [], [0, 0], [0, 0]
    pre_chunks = []
    slots = torch.empty(tab.cap, dtype=torch.long)
    bad = 0
    for s in streams:
        ids_p = tab.hash_tokens(s["prompt"], eos=EOS)
        hist = s["prompt"][-2:]
        ids_d = tab.hash_tokens(s["completion"], history=hist, eos=EOS)
        for a in range(0, ids_p.shape[0], chunk):
            ids = ids_p[a:a + chunk].reshape(-1).contiguous()
            c0 = st.counters()
            t0 = time.perf_counter()
            st.resolve(ids, slots)
            dt = time.perf_counter() - t0
            c1 = st.counters()
            pre_ms.append(dt * 1e3)
            pre_chunks.append(dict(name=s["name"], tokens=int(ids.numel() // 16), ms=dt * 1e3, lookups=int(ids.numel()),
                                   unique=c1[2] - c0[2], misses=c1[4] - c0[4], runs=c1[5] - c0[5], blocks=c1[7] - c0[7]))
            pre_hits[0] += c1[1] - c0[1]
            pre_hits[1] += c1[4] - c0[4]
            if check is not None:
                bad += verify(tab, ids, slots, check)
        for i in range(ids_d.shape[0]):
            ids = ids_d[i].contiguous()
            c0 = st.counters()
            t0 = time.perf_counter()
            st.resolve(ids, slots)
            dt = time.perf_counter() - t0
            c1 = st.counters()
            dec_us.append(dt * 1e6)
            dec_hits[0] += c1[1] - c0[1]
            dec_hits[1] += c1[4] - c0[4]
            if check is not None and (i % 7 == 0):
                bad += verify(tab, ids, slots, check)
    return dict(pre_ms=pre_ms, dec_us=dec_us, pre_chunks=pre_chunks,
                prefill_lookup_hit=1 - pre_hits[1] / max(pre_hits[0], 1),
                decode_lookup_hit=1 - dec_hits[1] / max(dec_hits[0], 1), bad_rows=bad, counters=tab.stats(gpu=False))


def verify(tab, ids, slots, rows_mm):
    n = ids.numel()
    got = torch.empty(n * tab.row_bytes, dtype=torch.uint8)
    tab.store.read_slots(slots[:n].contiguous(), got)
    got = got.numpy().reshape(n, tab.row_bytes)
    ref = rows_mm[ids.numpy()]
    return int((got != ref).any(axis=1).sum())


def file_rows(model):
    """Byte view of all rows (buffered mmap, independent of the store). Shards are contiguous in the file."""
    from sglang_exl3.offload.ngram_nvme import parse_table
    t = parse_table(model)
    rb = t["words"] * 2
    for i in range(1, len(t["offs"])):
        assert t["offs"][i] == t["offs"][i - 1] + t["rows"][i - 1] * rb, "shards not contiguous"
    return np.memmap(t["path"], dtype=np.uint8, mode="r", offset=t["offs"][0], shape=(t["num_rows"], rb))


def summarize(r, chunk):
    pc = [c for c in r["pre_chunks"] if c["tokens"] == chunk]
    d = r["dec_us"]
    return dict(
        chunk=chunk,
        prefill_chunks=len(r["pre_ms"]), prefill_full_chunks=len(pc),
        prefill_full_chunk_ms_mean=float(np.mean([c["ms"] for c in pc])) if pc else None,
        prefill_full_chunk_ms_max=float(np.max([c["ms"] for c in pc])) if pc else None,
        prefill_ms_per_1k_tok=1e3 * sum(r["pre_ms"]) / max(sum(c["tokens"] for c in r["pre_chunks"]), 1),
        prefill_lookup_hit=r["prefill_lookup_hit"],
        prefill_unique_frac=sum(c["unique"] for c in r["pre_chunks"]) / max(sum(c["lookups"] for c in r["pre_chunks"]), 1),
        prefill_blocks_per_miss=sum(c["blocks"] for c in r["pre_chunks"]) / max(sum(c["misses"] for c in r["pre_chunks"]), 1),
        decode_tokens=len(d), decode_lookup_hit=r["decode_lookup_hit"],
        decode_us_mean=float(np.mean(d)), decode_us_p50=pct(d, 50), decode_us_p90=pct(d, 90), decode_us_p99=pct(d, 99),
        decode_us_max=float(np.max(d)),
        decode_frac_all_hit=float(np.mean(np.asarray(d) < 20)),
        bad_rows=r["bad_rows"], counters=r["counters"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workloads", default="/w/kernels/cuda_sm86/traces/workloads.json")
    ap.add_argument("--k02", default="/w/runs/2026-09-29-K02-traces")
    ap.add_argument("--mode", default="sweep")
    ap.add_argument("--budgets", default="0.03,0.064,0.128,0.256,0.5,1,4")
    ap.add_argument("--io", default="aio")
    ap.add_argument("--chunks", default="8192,16384")
    ap.add_argument("--repeat", type=int, default=1, help="replay the streams this many times (warm-cache view)")
    ap.add_argument("--corpus", default="", help="hitrate mode: comma list of dirs; *.py/*.md/*.txt/*.rst files are "
                    "tokenised as warm-up traffic before the K02 streams")
    ap.add_argument("--io-cfgs", default="pread,32,warm")
    ap.add_argument("--policy", type=int, default=0, help="0 CLOCK, 1 GCLOCK (2-bit), 2 CLOCK cold insertion")
    ap.add_argument("--prefill-tps", type=float, default=2500.0)
    ap.add_argument("--corpus-tokens", type=int, default=20_000_000)
    ap.add_argument("--corpus-cache", default="")
    ap.add_argument("--drop", action="store_true", help="sweep: drop the n-gram file from the page cache first")
    a = ap.parse_args()
    os.environ["SGLANG_EXL3_NGRAM_POLICY"] = ["clock", "gclock", "cold"][a.policy]
    os.makedirs(a.out, exist_ok=True)
    streams = build_streams(a.model, a.workloads, a.k02)
    info = dict(requests=len(streams), prompt_tokens=sum(len(s["prompt"]) for s in streams),
                completion_tokens=sum(len(s["completion"]) for s in streams))
    print("streams", info, flush=True)
    res = dict(info=info, mode=a.mode, io=a.io)
    if a.mode == "hashcheck":
        tab = make_table(a.model, 0.03, a.io, 16384)
        fh = tab.file_hash
        bad = tot = 0
        for s in streams:
            toks = list(s["prompt"]) + list(s["completion"])
            # sprinkle eos to exercise segmentation
            toks = [EOS if (i % 997) == 5 else x for i, x in enumerate(toks)]
            r = ref_hash(toks, fh["layer_multipliers"], fh["head_vocab_sizes"], fh["head_offsets"], 8)
            c = tab.hash_tokens(toks, eos=EOS)
            bad += int((r != c).any(-1).sum()); tot += len(toks)
        res.update(hash_positions=tot, hash_mismatch=bad)
        print(res, flush=True)
    elif a.mode == "correct":
        rows = file_rows(a.model)
        out = []
        for gb in [float(x) for x in a.budgets.split(",")]:
            tab = make_table(a.model, gb, a.io, 16384)
            r = replay(tab, streams, 16384, check=rows)
            s = summarize(r, 16384); s["budget_gb"] = gb
            out.append(s)
            print(json.dumps({k: v for k, v in s.items() if k != "counters"}), flush=True)
            tab.release()
        # random ids incl. first/last row and the file tail
        tab = make_table(a.model, 0.5, a.io, 16384)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, tab.num_rows, (200000,), generator=g)
        ids[:4] = torch.tensor([0, 1, tab.num_rows - 2, tab.num_rows - 1])
        slots = torch.empty(tab.cap, dtype=torch.long)
        bad = 0
        for a0 in range(0, ids.numel(), 65536):
            x = ids[a0:a0 + 65536].contiguous()
            tab.store.resolve(x, slots)
            bad += verify(tab, x, slots, rows)
        res.update(replays=out, random_ids=int(ids.numel()), random_bad=bad)
        tab.release()
    elif a.mode == "hitrate":
        # dry cache simulation (no reads): warm-up traffic from a text/code corpus, then the K02 streams
        cache_f = a.corpus_cache
        if cache_f and os.path.exists(cache_f):
            z = np.load(cache_f)
            flat, offs = z["tokens"], z["offs"]
            docs = [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
            ntok = int(offs[-1])
        else:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
            files = []
            for d in filter(None, a.corpus.split(",")):
                for root, _, fs in os.walk(d):
                    for f in sorted(fs):
                        if f.endswith((".py", ".md", ".txt", ".rst")):
                            files.append(os.path.join(root, f))
            rng = np.random.default_rng(0)
            rng.shuffle(files)
            docs, ntok, B = [], 0, 256
            for i in range(0, len(files), B):
                texts = []
                for f in files[i:i + B]:
                    try:
                        texts.append(open(f, errors="ignore").read()[:400000])
                    except OSError:
                        pass
                for e in tok.encode_batch(texts, add_special_tokens=False):
                    if len(e.ids) < 16:
                        continue
                    docs.append(np.asarray(e.ids, dtype=np.int32))
                    ntok += len(e.ids)
                if ntok >= a.corpus_tokens:
                    break
            if cache_f:
                offs = np.cumsum([0] + [len(d) for d in docs])
                np.savez(cache_f, tokens=np.concatenate(docs), offs=offs)
        print("corpus", len(docs), "docs", ntok, "tokens", flush=True)
        out = []
        for gb in [float(x) for x in a.budgets.split(",")]:
            tab = make_table(a.model, gb, a.io, 16384)
            tab.store.set_dry(True)
            slots = torch.empty(tab.cap, dtype=torch.long)
            t0 = time.time()
            marks = {}
            seen = 0
            for dt in docs:
                d = tab.hash_tokens(torch.from_numpy(dt.astype(np.int64)), eos=EOS).reshape(-1)
                for c0 in range(0, d.numel(), tab.cap):
                    tab.store.resolve(d[c0:c0 + tab.cap].contiguous(), slots)
                seen += d.numel() // 16
                for m in (1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000, 150_000_000):
                    if seen >= m and m not in marks:
                        marks[m] = tab.stats(gpu=False)["lookup_hit_rate"]
            corpus_hit = tab.stats(gpu=False)["lookup_hit_rate"]
            tab.reset_stats()
            r = replay(tab, streams, 16384)
            o = dict(budget_gb=gb, slots=tab.nslots, corpus_tokens=ntok, corpus_lookup_hit=corpus_hit,
                     corpus_hit_marks=marks, k02_prefill_hit=r["prefill_lookup_hit"],
                     k02_decode_hit=r["decode_lookup_hit"], resident=tab.stats(gpu=False)["resident"],
                     sim_s=time.time() - t0)
            out.append(o)
            print(json.dumps(o), flush=True)
            tab.release()
        res.update(hitrate=out)
    elif a.mode == "hint":
        # prefill with a schedule-time hint: chunk i+1 is warmed asynchronously while chunk i "computes"
        # (sleep = chunk tokens / --prefill-tps), then chunk i+1's lookup is timed. Cold = no hint, same cache start.
        out = []
        for chunk in [int(x) for x in a.chunks.split(",")]:
            for hint in (False, True):
                tab = make_table(a.model, 4.0, a.io, chunk)
                slots = torch.empty(tab.cap, dtype=torch.long)
                for s_ in streams:
                    if s_["cat"] != "long":
                        continue
                    toks = s_["prompt"]
                    ids_p = tab.hash_tokens(toks, eos=EOS)
                    n = ids_p.shape[0]
                    for c0 in range(0, n, chunk):
                        x = ids_p[c0:c0 + chunk].reshape(-1).contiguous()
                        t0 = time.perf_counter(); tab.store.resolve(x, slots); dt = time.perf_counter() - t0
                        fut = None
                        if hint and c0 + chunk < n:
                            fut = tab.hint_tokens(toks[c0 + chunk:c0 + 2 * chunk], history=toks[c0 + chunk - 2:c0 + chunk], eos=EOS)
                        time.sleep(min(chunk, n - c0) / a.prefill_tps)
                        warm_done = fut.done() if fut is not None else None
                        out.append(dict(chunk=chunk, hint=hint, name=s_["name"], c0=c0, tokens=int(x.numel() // 16),
                                        ms=dt * 1e3, hint_done_before_next=warm_done))
                        print(json.dumps(out[-1]), flush=True)
                tab.release()
        res.update(hint=out)
    elif a.mode == "io":
        # IO mode study: cold random rows (fresh ids every rep) through the store, per backend/threads, with the page
        # cache warm (as found) and cold (per-file POSIX_FADV_DONTNEED of the n-gram file only). dm-0 read counters
        # before/after flag foreign IO (the plugin's Flash-Next server shares the NVMe).
        def dm_reads():
            try:
                f = open("/sys/block/dm-0/stat").read().split()
                return int(f[0]), int(f[2]) * 512
            except Exception:
                return 0, 0
        out = []
        g = torch.Generator().manual_seed(7)
        for cfg in a.io_cfgs.split(";"):
            io, thr, cache = cfg.split(",")[:3]
            os.environ["SGLANG_EXL3_NGRAM_THREADS"] = thr
            os.environ["SGLANG_EXL3_NGRAM_PC_DROP"] = "1" if cfg.endswith(",drop") else "0"
            tab = make_table(a.model, 1.0, io, 16384)
            if cache == "cold":
                tab.store.drop_file_cache()
            slots = torch.empty(tab.cap, dtype=torch.long)
            for n in (16, 4096, 131072, 262144):
                reps = {16: 300, 4096: 20, 131072: 3, 262144: 2}[n]
                ts = []
                d0 = dm_reads()
                c0 = tab.stats(gpu=False)
                for _ in range(reps):
                    x = torch.randint(0, tab.num_rows, (n,), generator=g)
                    t0 = time.perf_counter(); tab.store.resolve(x, slots); ts.append(time.perf_counter() - t0)
                d1 = dm_reads()
                c1 = tab.stats(gpu=False)
                own = c1["bytes"] - c0["bytes"]
                o = dict(io=io, threads=int(thr), page_cache=cache, pc_drop=cfg.endswith(",drop"), n=n, reps=reps, ms_p50=1e3 * pct(ts, 50),
                         ms_mean=1e3 * float(np.mean(ts)), ms_max=1e3 * float(np.max(ts)),
                         rows_per_s=n / float(np.median(ts)), own_bytes=own, dm0_read_bytes=d1[1] - d0[1],
                         dm0_read_ios=d1[0] - d0[0])
                out.append(o)
                print(json.dumps(o), flush=True)
            tab.release()
        res.update(io=out)
    elif a.mode == "qd":
        # cold random reads through the store: n ids per resolve, fresh rows each time
        out = []
        tab = make_table(a.model, 4.0, a.io, 16384)
        g = torch.Generator().manual_seed(1)
        slots = torch.empty(tab.cap, dtype=torch.long)
        for n in (16, 64, 256, 1024, 4096, 16384, 65536):
            reps = max(3, min(400, 200000 // n))
            ts = []
            for _ in range(reps):
                x = torch.randint(0, tab.num_rows, (n,), generator=g)
                t0 = time.perf_counter(); tab.store.resolve(x, slots); ts.append(time.perf_counter() - t0)
            c = tab.stats(gpu=False)
            out.append(dict(n=n, reps=reps, ms_mean=1e3 * float(np.mean(ts)), ms_p50=1e3 * pct(ts, 50),
                            ms_p99=1e3 * pct(ts, 99), iops=n / float(np.mean(ts))))
            print(json.dumps(out[-1]), flush=True)
        res.update(qd=out)
        tab.release()
    else:
        out = []
        for chunk in [int(x) for x in a.chunks.split(",")]:
            for gb in [float(x) for x in a.budgets.split(",")]:
                tab = make_table(a.model, gb, a.io, chunk)
                if a.drop:
                    tab.store.drop_file_cache()
                for rep in range(a.repeat):
                    r = replay(tab, streams, chunk)
                    s = summarize(r, chunk); s.update(budget_gb=gb, rep=rep, io=a.io)
                    tab.reset_stats()
                    out.append(s)
                    print(json.dumps({k: v for k, v in s.items() if k != "counters"}), flush=True)
                if a.repeat == 1:
                    np.save(os.path.join(a.out, f"dec_us_{chunk}_{gb}.npy"), np.asarray(r["dec_us"], dtype=np.float32))
                    json.dump(r["pre_chunks"], open(os.path.join(a.out, f"pre_chunks_{chunk}_{gb}.json"), "w"))
                tab.release()
        res.update(sweep=out)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
