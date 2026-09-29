"""
Prefill / decode speed against an SGLang server (C1).
  prefill N : teacher-forced scoring request (max_new_tokens=0) of N prompt tokens -> tok/s (no generation involved)
  decode    : greedy chat 'count from 1 to 200' (thinking off, natural stop, no length cap) after an optional filler
              context of C tokens; decode tok/s = (completion tokens - 1) / (t_last - t_first) from the stream
  python3 speed_client.py --url http://127.0.0.1:30200 --prefill 4096,32768 --decode-ctx 0,32768 --out r.json
"""
import argparse, json, random, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30200")
ap.add_argument("--prefill", default="4096")
ap.add_argument("--decode-ctx", default="0")
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--out", default="")
a = ap.parse_args()


def post(path, body, timeout=7200):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def filler_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


res = {"prefill": [], "decode": []}
for n in [int(v) for v in a.prefill.split(",") if v]:
    for rep in range(a.reps):
        ids = filler_ids(n, 1000 * n + rep)
        t0 = time.time()
        post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 0, "temperature": 0}})
        dt = time.time() - t0
        r = {"tokens": n, "rep": rep, "seconds": round(dt, 3), "tok_per_s": round(n / dt, 1)}
        res["prefill"].append(r)
        print(r, flush=True)

tok = None
for ctx in [int(v) for v in a.decode_ctx.split(",") if v != ""]:
    for rep in range(a.reps):
        msgs = []
        if ctx:
            rng = random.Random(ctx + rep)          # "itemNNNNNN" = ~7 tokens per word with this tokenizer (measured)
            words = " ".join(f"item{rng.randrange(10**6)}" for _ in range(int(ctx / 7.0)))
            msgs.append({"role": "user", "content": "Here is a long log you can ignore:\n" + words})
            msgs.append({"role": "assistant", "content": "Understood."})
        msgs.append({"role": "user", "content": "Count from 1 to 200, separated by commas. Output only the numbers."})
        body = {"model": "flashnext", "messages": msgs, "temperature": 0, "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time(); t_first = None; t_last = None; n_chunks = 0; usage = None
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
                    n_chunks += 1
        comp = usage["completion_tokens"] if usage else n_chunks
        r = {"ctx_tokens": usage["prompt_tokens"] if usage else None, "rep": rep, "completion_tokens": comp,
             "ttft_s": round(t_first - t0, 2) if t_first else None,
             "decode_tok_per_s": round((comp - 1) / (t_last - t_first), 2) if t_first and t_last > t_first else None}
        res["decode"].append(r)
        print(r, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
