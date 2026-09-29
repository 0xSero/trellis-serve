# K02: capture real routing traces (per-layer top-10 expert ids + router weights) from stock exllamav3 1.5.1 on
# Qwen3.8-Flash-Next (experts on CPU, -mcl 48). Hook: every BlockSparseMLP instance's routing_fn is wrapped; each call
# (bsz rows) is cloned on the GPU (ids int16, weights fp16) and flushed per request.
# Output per request: <out>/<name>.npz with
#   prefill_ids  int16 [P, 48, 10], prefill_w fp16 [P, 48, 10]  (rows of routing calls with bsz > 1, in order)
#   decode_ids   int16 [D, 48, 10], decode_w  fp16 [D, 48, 10]  (bsz == 1 calls = one generated token each)
#   prompt_len, completion_tokens, text (completion), think flag
# Generation runs to EOS (no max_new_tokens cap: the cache size is the only bound).
import sys, os, re, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, "/opt/exllamav3-src")
from exllamav3 import model_init, Generator, ComboSampler

REC = []          # (layer, ids_gpu int16 [bsz,10], w_gpu fp16 [bsz,10])


def walk(m, seen):
    if id(m) in seen:
        return
    seen.add(id(m))
    yield m
    for s in getattr(m, "modules", []) or []:
        yield from walk(s, seen)


def install_hooks(model):
    n = 0
    for m in walk(model, set()):
        if type(m).__name__ == "BlockSparseMLP" and getattr(m, "routing_gate", None) is not None:
            mt = re.search(r"layers\.(\d+)\.", m.key)
            if mt is None or m.key.startswith("mtp"):
                continue
            layer = int(mt.group(1))
            orig = m.routing_fn

            def hooked(bsz, cfg, z, params, _orig=orig, _layer=layer):
                sel, w = _orig(bsz, cfg, z, params)
                REC.append((_layer, sel.to(torch.int16).clone(), w.to(torch.float16).clone()))
                return sel, w
            m.routing_fn = hooked
            n += 1
    return n


def flush(num_layers):
    """REC -> (prefill [P, L, k], decode [D, L, k]) numpy arrays. Calls arrive layer 0..L-1 per forward."""
    torch.cuda.synchronize()
    seq = [(g[0], int(g[1].shape[0])) for g in REC]
    if os.environ.get("TRACE_DEBUG"):
        print("CALLSEQ", len(seq), seq[:150], "...", seq[-60:], flush=True)
    pre_i, pre_w, dec_i, dec_w = [], [], [], []
    # one forward = consecutive calls with strictly increasing layer and the same bsz (a prefill forward skips the last
    # layer's MoE: it only feeds the logits); layers a forward did not route are stored as id -1 / weight 0
    fwds, cur = [], []
    for g in REC:
        if cur and (g[0] <= cur[-1][0] or g[1].shape[0] != cur[-1][1].shape[0]):
            fwds.append(cur); cur = []
        cur.append(g)
    if cur:
        fwds.append(cur)
    for f in fwds:
        bsz = f[0][1].shape[0]
        ids = torch.full((bsz, num_layers, 10), -1, dtype=torch.int16, device=f[0][1].device)
        w = torch.zeros((bsz, num_layers, 10), dtype=torch.float16, device=f[0][1].device)
        for layer, gi, gw in f:
            ids[:, layer] = gi; w[:, layer] = gw
        (dec_i if bsz == 1 else pre_i).append(ids.cpu().numpy())
        (dec_w if bsz == 1 else pre_w).append(w.cpu().numpy())
    REC.clear()
    k = 10
    cat = lambda xs, dt: np.concatenate(xs, 0).astype(dt) if xs else np.zeros((0, num_layers, k), dt)
    return cat(pre_i, np.int16), cat(pre_w, np.float16), cat(dec_i, np.int16), cat(dec_w, np.float16)


def main():
    p = argparse.ArgumentParser()
    model_init.add_args(p, cache=True)
    p.add_argument("--workloads", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model, config, cache, tok = model_init.init(args)
    nl = install_hooks(model)
    print("hooked BlockSparseMLP layers:", nl, flush=True)
    REC.clear()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    stop = [tok.eos_token_id] + [tok.single_id(s) for s in ("<|im_end|>", "<|endoftext|>") if tok.single_id(s) is not None]
    sampler = ComboSampler(temperature=0.6, top_p=0.95, top_k=20)     # model-card style thinking-mode sampling
    W = json.load(open(args.workloads))
    readme = open(os.path.join(args.model_dir, "README.md")).read()
    tasks = [(cat, t) for cat in ("prose", "code", "long") for t in W[cat]]
    only = set(filter(None, args.only.split(",")))
    summary = []
    for cat, t in tasks:
        if only and t["name"] not in only and cat not in only:
            continue
        if os.path.exists(os.path.join(args.out, t["name"] + ".npz")):
            print("skip (done)", t["name"], flush=True)
            continue
        doc = readme if t.get("doc_file") == "MODEL_README" else t.get("doc", "")
        q = (doc + "\n\n" + t["question"]) if doc else t["question"]
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        if not t["think"]:
            text += "<think>\n\n</think>\n\n"
        ids = tok.encode(text, encode_special_tokens=True)
        plen = ids.shape[-1]
        REC.clear()
        t0 = time.time()
        comp = gen.generate(prompt=text, stop_conditions=stop, completion_only=True, encode_special_tokens=True,
                            sampler=sampler, seed=1234, max_new_tokens=args.cache_size - plen - 256)
        dt = time.time() - t0
        pi, pw, di, dw = flush(nl)
        ctoks = tok.encode(comp, encode_special_tokens=True).shape[-1]
        np.savez_compressed(os.path.join(args.out, t["name"] + ".npz"), prefill_ids=pi, prefill_w=pw, decode_ids=di,
                            decode_w=dw, prompt_len=plen, completion_tokens=ctoks, think=t["think"], category=cat,
                            text=np.array(comp))
        rec = {"name": t["name"], "category": cat, "think": t["think"], "prompt_len": plen, "prefill_rows": int(pi.shape[0]),
               "decode_steps": int(di.shape[0]), "completion_tokens": int(ctoks), "seconds": round(dt, 1),
               "tok_per_s": round(di.shape[0] / dt, 2) if dt else None}
        summary.append(rec)
        print("TRACE", json.dumps(rec), "|", comp[:120].replace("\n", " "), flush=True)
        with open(os.path.join(args.out, "summary.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    print("TRACES_DONE", len(summary), flush=True)


if __name__ == "__main__":
    main()
