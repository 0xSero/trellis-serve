"""
Correctness gate against the exllamav3 reference panel (runs/2026-09-29-E003b-ref-panel/ref_panel.json):
teacher-force every panel sequence through an SGLang server (/generate, input_ids, max_new_tokens=0, input top-k
logprobs) and compare the next-token distributions from position prompt_len-1 on with the reference top-20:
  - top-1 agreement (our argmax == reference argmax)
  - KL(ref || ours) on the reference top-20 support (both renormalised over those 20 ids; ids missing from our top-K
    get our smallest returned logprob, an upper bound on their mass)
Also greedy chat: each prompt generated greedily (no length cap) and compared with the reference completion
(common token prefix, exact match).

  python3 ref_panel_client.py --panel ref_panel.json --url http://127.0.0.1:30200 --out result.json [--no-greedy]
"""
import argparse, json, math, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--panel", required=True)
ap.add_argument("--url", default="http://127.0.0.1:30200")
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--no-greedy", action="store_true")
ap.add_argument("--out", default="")
ap.add_argument("--ctx", type=int, default=262144)
a = ap.parse_args()


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


panel = json.load(open(a.panel))
res = {"items": []}
agree = n = 0
kls = []
for i, it in enumerate(panel):
    toks, pl = it["tokens"], it["prompt_len"]
    t0 = time.time()
    r = post("/generate", {"input_ids": toks, "sampling_params": {"max_new_tokens": 0, "temperature": 0},
                           "return_logprob": True, "logprob_start_len": pl - 1, "top_logprobs_num": a.topk})
    dt = time.time() - t0
    tops = r["meta_info"]["input_top_logprobs"]
    # tops[j] belongs to input token start+j (distribution that predicted it); tops[0] (the start token) may be None.
    # ref top20[j] = distribution after tokens[:pl+j] (predicts tokens[pl+j]) -> ours = tops[j+1]
    ours = [t for t in tops]
    off = 1
    m_agree = m_n = 0
    m_kl = []
    for j, (rid, rlp) in enumerate(zip(it["top20_ids"], it["top20_lp"])):
        if j + off >= len(ours) or ours[j + off] is None:
            continue
        o = {int(tid): lp for lp, tid, *_ in ours[j + off]}
        o_min = min(o.values())
        o_top = max(o, key=o.get)
        m_agree += int(o_top == rid[0]); m_n += 1
        p = [math.exp(x) for x in rlp]; zp = sum(p); p = [x / zp for x in p]
        q = [math.exp(o.get(t, o_min)) for t in rid]; zq = sum(q); q = [x / zq for x in q]
        m_kl.append(sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q) if pi > 0))
    agree += m_agree; n += m_n; kls += m_kl
    rec = {"i": i, "positions": m_n, "top1_agree": round(m_agree / max(m_n, 1), 4),
           "kl_top20_mean": round(sum(m_kl) / max(len(m_kl), 1), 5), "prefill_s": round(dt, 2), "tokens": len(toks)}
    # sanity of the alignment: the reference's own greedy continuation
    rec["ref_top1_is_next_token"] = round(sum(int(it["top20_ids"][j][0] == toks[pl + j]) for j in range(len(toks) - pl)) /
                                          max(len(toks) - pl, 1), 3)
    res["items"].append(rec)
    print(rec, flush=True)
res["top1_agreement"] = round(agree / max(n, 1), 4)
res["kl_top20_mean"] = round(sum(kls) / max(len(kls), 1), 5)
kls.sort()
res["kl_top20_p50"] = round(kls[len(kls) // 2], 5) if kls else None
res["kl_top20_p95"] = round(kls[int(len(kls) * 0.95)], 5) if kls else None
res["positions"] = n
print({k: v for k, v in res.items() if k != "items"}, flush=True)

if not a.no_greedy:
    res["greedy"] = []
    for i, it in enumerate(panel):
        toks, pl = it["tokens"], it["prompt_len"]
        t0 = time.time()
        # SGLang's /generate defaults to max_new_tokens=128: pass the remaining context so generation ends naturally
        r = post("/generate", {"input_ids": toks[:pl], "sampling_params": {"temperature": 0,
                                                                           "max_new_tokens": a.ctx - pl}})
        dt = time.time() - t0
        out = r.get("output_ids") or []
        ref = toks[pl:]
        common = 0
        for x, y in zip(out, ref):
            if x != y:
                break
            common += 1
        g = {"i": i, "gen_tokens": len(out), "ref_tokens": len(ref), "common_prefix": common, "exact": out == ref,
             "seconds": round(dt, 1), "finish": (r.get("meta_info") or {}).get("finish_reason"),
             "text_head": r.get("text", "")[:160], "text": r.get("text", "")}
        res["greedy"].append(g)
        print({k: v for k, v in g.items() if k != "text"}, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
