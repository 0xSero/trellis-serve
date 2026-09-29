#!/usr/bin/env python3
"""K08: torch-profile the prefill chunks of one long prompt on a running SGLang server and break GPU time down.
  python3 profile_prefill.py --url http://127.0.0.1:30200 --ctx 32768 --steps 4 --dir /w/runs/X/prof --host-dir ~/freetoken-exl3/runs/X/prof
The request streams and is dropped after its first generated token (no max_tokens cap is set)."""
import argparse, glob, gzip, json, os, random, time, urllib.request
from collections import defaultdict

CATS = [  # (category, substrings) first match wins
    ("moe_grouped_gemm", ["trellis_exl3_marlin_moe::Marlin"]),
    ("moe_small (had/glu/combine/align/cache/copy)", ["trellis_moe_", "moe_align", "count_and_sort", "trellis_cache_", "copy_records", "moe_combine"]),
    ("dense_exl3_marlin", ["trellis_exl3_marlin::Marlin", "exl3_gemv", "exl3_gemm", "exl3_mgemm"]),
    ("dense_reconstruct+cublas", ["reconstruct", "ampere_", "sm80_xmma", "cutlass", "gemm", "gemv"]),
    ("hadamard", ["had_r_128", "had_hf", "had_"]),
    ("hc_mix/combine/norm", ["hc_mix", "hc_combine", "grouped_gemma_rmsnorm", "_mix_compute", "triton_poi", "triton_red", "triton_per"]),
    ("gdn", ["chunk_", "gated_delta", "fused_recurrent", "causal_conv1d", "l2norm", "solve_tril", "recompute_w_u", "_layer_norm"]),
    ("qsa/attention", ["qsa", "flash", "attn", "_compact_kv", "indexer", "topk", "fast_topk", "sparse"]),
    ("ngram/ple", ["ngram", "ple", "embedding", "gather"]),
    ("memcpy/memset", ["Memcpy", "Memset", "memcpy", "memset"]),
]


def cat_of(name):
    for c, keys in CATS:
        if any(k in name for k in keys):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30200")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--host-dir", required=True)
    ap.add_argument("--no-profile", action="store_true", help="just time the prefill (TTFT)")
    a = ap.parse_args()
    rng = random.Random(time.time_ns())      # unique prompt per request: no radix prefix-cache hits
    words = open("/usr/share/dict/words").read().split() if os.path.exists("/usr/share/dict/words") else None
    ids = [rng.randrange(1000, 150000) for _ in range(a.ctx)]
    if not a.no_profile:
        req = urllib.request.Request(a.url + "/start_profile", data=json.dumps(
            {"output_dir": a.dir, "num_steps": a.steps, "activities": ["CPU", "GPU"]}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
    body = {"input_ids": ids, "sampling_params": {"temperature": 0}, "stream": True}
    t0 = time.time()
    req = urllib.request.Request(a.url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    ttft = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data:"):
                ttft = time.time() - t0
                break
    print(f"TTFT {ttft:.2f} s for {a.ctx} tokens -> {a.ctx / ttft:.0f} tok/s (wall, incl. profiler overhead if on)", flush=True)
    if a.no_profile:
        return
    time.sleep(8)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.host_dir), "*.json*")), key=os.path.getmtime)
    if not files:
        print("no trace written"); return
    f = files[-1]
    data = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
    ev = data["traceEvents"] if isinstance(data, dict) else data
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    agg, cnt, cat = defaultdict(float), defaultdict(int), defaultdict(float)
    streams = defaultdict(float)
    for e in kern:
        agg[e["name"]] += e.get("dur", 0); cnt[e["name"]] += 1
        cat[cat_of(e["name"])] += e.get("dur", 0)
        streams[e.get("args", {}).get("stream", "?")] += e.get("dur", 0)
    t_first = min(e["ts"] for e in kern); t_last = max(e["ts"] + e.get("dur", 0) for e in kern)
    # busy time on the compute stream(s) excluding the copy stream: union of intervals of non-memcpy kernels
    iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in kern if e.get("cat") == "kernel")
    busy, cur_s, cur_e = 0.0, None, None
    for s, t in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, t
        else:
            cur_e = max(cur_e, t)
    busy += (cur_e - cur_s) if cur_e else 0
    wall = t_last - t_first
    tot = sum(agg.values())
    print(f"trace {f}: {a.steps} steps, GPU window {wall / 1e3:.1f} ms, compute-kernel busy (union) {busy / 1e3:.1f} ms "
          f"(idle {100 * (1 - busy / wall):.1f}%), summed kernel+copy time {tot / 1e3:.1f} ms", flush=True)
    print("BY CATEGORY (ms per step):")
    for c, d in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"  {d / 1e3 / a.steps:9.2f} ms  {100 * d / tot:5.1f}%  {c}")
    print("TOP KERNELS (ms per step):")
    for name, d in sorted(agg.items(), key=lambda kv: -kv[1])[:40]:
        print(f"  {d / 1e3 / a.steps:9.2f} ms {cnt[name] / a.steps:7.1f}x  [{cat_of(name)}] {name[:140]}")
    print("STREAMS (ms per step):", {k: round(v / 1e3 / a.steps, 2) for k, v in streams.items()})


if __name__ == "__main__":
    main()
