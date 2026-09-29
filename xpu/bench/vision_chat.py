"""One image + question through the OpenAI chat endpoint (greedy, no length cap). Prints answer, usage, latency.
  python3 vision_chat.py --url http://127.0.0.1:30350 IMAGE "question" [IMAGE "question" ...]"""
import argparse, base64, json, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30350")
ap.add_argument("pairs", nargs="+")
a = ap.parse_args()
for img, q in zip(a.pairs[::2], a.pairs[1::2]):
    ext = os.path.splitext(img)[1].lstrip(".").replace("jpg", "jpeg")
    url = f"data:image/{ext};base64," + base64.b64encode(open(img, "rb").read()).decode()
    body = {"model": "flashnext", "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": q}]}],
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.load(r)
    dt = time.time() - t0
    print(json.dumps({"image": os.path.basename(img), "seconds": round(dt, 2), "usage": out.get("usage"),
                      "answer": out["choices"][0]["message"]["content"]}, ensure_ascii=False), flush=True)
