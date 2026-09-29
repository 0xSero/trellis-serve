"""Vision gate against an exllamav3 multimodal reference panel (vision/tools/ref_vision.py output).

  tf     : teacher-force every reference sequence (prompt text with one <|image_pad|> expanded server-side + the
           reference completion) through SGLang /generate with image_data; compare next-token distributions over the
           completion positions with the reference top-20: top-1 agreement, KL(ref || ours) on the top-20 support
           (same definition as ref_panel_client.py). Alignment is checked token by token (SGLang's expanded input ids
           vs the reference's, image positions included) before scoring.
  greedy : the same prompt, greedy, no length cap (natural EOS); text + prefix match vs the reference completion.
  probes : size probes (256^2 .. 4096^2 images carrying a code): latency, prompt tokens, answer, code read back.

  python3 vision_gate.py --ref ref_vision.json --images <dir> --url http://127.0.0.1:30350 --out gate.json
         [--no-greedy] [--probes] [--only-probes]
"""
import argparse, base64, json, math, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="")
ap.add_argument("--images", required=True)
ap.add_argument("--url", default="http://127.0.0.1:30350")
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--ctx", type=int, default=262144)
ap.add_argument("--no-greedy", action="store_true")
ap.add_argument("--no-tf", action="store_true")
ap.add_argument("--probes", action="store_true")
ap.add_argument("--probe-sizes", default="")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--out", default="")
a = ap.parse_args()


def post(path, body, timeout=7200):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def data_url(path):
    ext = os.path.splitext(path)[1].lstrip(".").replace("jpg", "jpeg")
    return f"data:image/{ext};base64," + base64.b64encode(open(path, "rb").read()).decode()


TEMPLATE = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
res = {"items": [], "greedy": [], "probes": []}

if a.ref and not a.no_tf:
    ref = json.load(open(a.ref))
    agree = n = 0
    kls = []
    for rep in range(a.repeat):
        for i, it in enumerate(ref):
            img = data_url(os.path.join(a.images, it["file"]))
            text = it.get("text_srv") or TEMPLATE.format(q=it["question"])
            pl, toks = it["prompt_len"], it["tokens"]
            t0 = time.time()
            r = post("/generate", {"text": text + it["completion"], "image_data": img,
                                   "sampling_params": {"max_new_tokens": 0, "temperature": 0},
                                   "return_logprob": True, "logprob_start_len": pl - 1, "top_logprobs_num": a.topk})
            dt = time.time() - t0
            mi = r["meta_info"]
            got = [t for _, t, *_ in mi["input_token_logprobs"]]
            # input_token_logprobs covers input positions logprob_start_len .. end
            want = toks[pl - 1:]
            mism = sum(1 for x, y in zip(got, want) if x != y)
            aligned = len(got) == len(want) and mism == 0
            tops = mi["input_top_logprobs"]
            m_agree = m_n = 0
            m_kl = []
            bad = []
            for j, (rid, rlp) in enumerate(zip(it["top20_ids"], it["top20_lp"])):
                if j + 1 >= len(tops) or tops[j + 1] is None:
                    continue
                o = {int(tid): lp for lp, tid, *_ in tops[j + 1]}
                o_min = min(o.values())
                o_top = max(o, key=o.get)
                m_agree += int(o_top == rid[0]); m_n += 1
                p = [math.exp(x) for x in rlp]; zp = sum(p); p = [x / zp for x in p]
                q = [math.exp(o.get(t, o_min)) for t in rid]; zq = sum(q); q = [x / zq for x in q]
                kl = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q) if pi > 0)
                m_kl.append(kl)
                if kl > 1:
                    bad.append((j, round(kl, 2)))
            agree += m_agree; n += m_n; kls += m_kl
            rec = {"rep": rep, "i": i, "name": it["name"], "n_image_tokens": it["n_image_tokens"],
                   "prompt_tokens": mi.get("prompt_tokens"), "aligned": aligned, "id_mismatches": mism,
                   "len_ours": len(got), "len_ref": len(want), "positions": m_n,
                   "top1_agree": round(m_agree / max(m_n, 1), 4), "kl_top20_mean": round(sum(m_kl) / max(len(m_kl), 1), 5),
                   "kl_gt1_positions": bad, "seconds": round(dt, 2)}
            res["items"].append(rec)
            print(rec, flush=True)
    res["top1_agreement"] = round(agree / max(n, 1), 4)
    res["kl_top20_mean"] = round(sum(kls) / max(len(kls), 1), 5)
    kls.sort()
    res["kl_top20_p50"] = round(kls[len(kls) // 2], 5) if kls else None
    res["kl_top20_p95"] = round(kls[int(len(kls) * 0.95)], 5) if kls else None
    res["positions"] = n
    print({k: v for k, v in res.items() if k not in ("items", "greedy", "probes")}, flush=True)

    if not a.no_greedy:
        for i, it in enumerate(ref):
            img = data_url(os.path.join(a.images, it["file"]))
            text = it.get("text_srv") or TEMPLATE.format(q=it["question"])
            t0 = time.time()
            r = post("/generate", {"text": text, "image_data": img,
                                   "sampling_params": {"temperature": 0, "max_new_tokens": a.ctx - it["prompt_len"]}})
            dt = time.time() - t0
            out = r.get("output_ids") or []
            refc = it.get("completion_ids") or it["tokens"][it["prompt_len"]:]
            common = 0
            for x, y in zip(out, refc):
                if x != y:
                    break
                common += 1
            g = {"i": i, "name": it["name"], "gen_tokens": len(out), "ref_tokens": len(refc), "common_prefix": common,
                 "exact": out == refc, "seconds": round(dt, 1), "finish": (r.get("meta_info") or {}).get("finish_reason"),
                 "text": r.get("text", ""), "ref_text": it["completion"]}
            res["greedy"].append(g)
            print({k: v for k, v in g.items() if k not in ("text", "ref_text")}, "|", g["text"][:300].replace("\n", " "),
                  flush=True)

if a.probes:
    meta = json.load(open(os.path.join(a.images, "images.json")))
    sizes = [int(s) for s in a.probe_sizes.split(",")] if a.probe_sizes else None
    for pr in meta.get("probes", []):
        if sizes and pr["size"] not in sizes:
            continue
        img = data_url(os.path.join(a.images, pr["file"]))
        q = pr.get("question") or "What is the code written in this image? Answer with the code only."
        t0 = time.time()
        try:
            r = post("/generate", {"text": TEMPLATE.format(q=q), "image_data": img,
                                   "sampling_params": {"temperature": 0, "max_new_tokens": a.ctx - 70000}})
            err = None
        except Exception as e:  # noqa: BLE001
            r, err = {}, repr(e)
        dt = time.time() - t0
        mi = r.get("meta_info") or {}
        txt = r.get("text", "")
        rec = {"size": pr["size"], "code": pr.get("code"), "prompt_tokens": mi.get("prompt_tokens"),
               "completion_tokens": mi.get("completion_tokens"), "e2e_s": round(dt, 2), "answer": txt[:200],
               "code_read": bool(pr.get("code")) and pr["code"] in txt.replace(" ", ""), "error": err}
        res["probes"].append(rec)
        print(rec, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
