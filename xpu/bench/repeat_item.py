"""
Determinism probe: teacher-force ONE ref-panel item N times against a running server and report, per run, the item's
mean KL(ref || ours) on the ref top-20 support plus the positions whose KL exceeds --hot, and the max |dlogprob| of our
top-1 vs run 0 (0.0 everywhere = bit-deterministic serving).

  python3 repeat_item.py --panel ref_panel.json --item 3 --n 8 --url http://127.0.0.1:30251 --out rep.json
"""
import argparse, json, math, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--panel", required=True)
ap.add_argument("--item", type=int, default=3)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--url", default="http://127.0.0.1:30250")
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--hot", type=float, default=0.05)
ap.add_argument("--flush", action="store_true", help="POST /flush_cache before every run (no radix prefix reuse)")
ap.add_argument("--out", default="")
a = ap.parse_args()


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        t = r.read()
        return json.loads(t) if t[:1] in b"{[" else t


it = json.load(open(a.panel))[a.item]
toks, pl = it["tokens"], it["prompt_len"]
runs, base = [], None
for r in range(a.n):
    if a.flush:
        post("/flush_cache", {})
    out = post("/generate", {"input_ids": toks, "sampling_params": {"max_new_tokens": 0, "temperature": 0},
                             "return_logprob": True, "logprob_start_len": pl - 1, "top_logprobs_num": a.topk})
    tops = out["meta_info"]["input_top_logprobs"]
    kl, top1 = [], []
    for j, (rid, rlp) in enumerate(zip(it["top20_ids"], it["top20_lp"])):
        if j + 1 >= len(tops) or tops[j + 1] is None:
            continue
        o = {int(tid): lp for lp, tid, *_ in tops[j + 1]}
        o_min = min(o.values())
        p = [math.exp(x) for x in rlp]; zp = sum(p); p = [x / zp for x in p]
        q = [math.exp(o.get(t, o_min)) for t in rid]; zq = sum(q); q = [x / zq for x in q]
        kl.append(sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q) if pi > 0))
        top1.append(max(o.values()))
    if base is None:
        base = top1
    dmax = max(abs(x - y) for x, y in zip(top1, base))
    first_diff = next((j for j, (x, y) in enumerate(zip(top1, base)) if x != y), None)
    rec = {"run": r, "kl_mean": round(sum(kl) / len(kl), 5), "hot": [(j, round(k, 3)) for j, k in enumerate(kl) if k > a.hot],
           "max_dlp_vs_run0": round(dmax, 5), "first_diff_pos": first_diff, "positions": len(kl)}
    runs.append(rec)
    print(rec, flush=True)
if a.out:
    json.dump({"item": a.item, "n": a.n, "flush": a.flush, "runs": runs}, open(a.out, "w"), indent=1)
