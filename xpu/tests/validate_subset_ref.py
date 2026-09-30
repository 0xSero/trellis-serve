"""One-time check that the stress test's SubsetRef equals the full pinned table (Exl3NgramHostTable) on rows [0, N).

Loads both (~35 GB pinned for a minute), compares gathers bit for bit on random ids including the range edges and
every shard boundary below N, frees the full table, and writes a JSON verdict. No NVMe tier, no atomics.
Exit 0 pass, 1 mismatch, 3 error.
"""
import argparse, json, os, re, sys, time, traceback
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/model")
ap.add_argument("--ref-rows", type=int, default=20_000_000)
ap.add_argument("--ids", type=int, default=1 << 20)
ap.add_argument("--out", required=True)
a = ap.parse_args()
res = dict(ref_rows=a.ref_rows, ids=a.ids, status="setup", started=time.time())
try:
    from exl3xpu.ngram_host import Exl3NgramHostTable, FILE, _header
    src = open(os.path.join(HERE, "test_ngram_nvme_stress.py")).read()
    ns = {}
    # take the SubsetRef class exactly as the stress test defines it (importing the test would run it)
    start = src.index("class SubsetRef:")
    end = src.index("\ndev = torch.device(", start)
    exec("import os, re, torch\nfrom exl3xpu.ngram_host import FILE, _header, _read_small\n" + src[start:end], ns)
    dev = torch.device("xpu", 0)
    sub = ns["SubsetRef"](a.model, a.ref_rows)
    full = Exl3NgramHostTable(a.model)
    hdr, _ = _header(os.path.join(a.model, FILE))
    rows_per = [v["shape"][0] for k, v in sorted(((k, v) for k, v in hdr.items() if k.endswith(".trellis")),
                                                 key=lambda kv: int(re.search(r"shard_(\d+)", kv[0]).group(1)))]
    edges, r = [0, 1, sub.num_rows - 2, sub.num_rows - 1], 0
    for n in rows_per:
        r += n
        if r < sub.num_rows:
            edges += [r - 1, r]
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, sub.num_rows, (a.ids,), generator=g, dtype=torch.int64)
    ids[:len(edges)] = torch.tensor(edges, dtype=torch.int64)
    ids = ids.view(-1, 16).to(dev)
    x, y = sub.gather(ids), full.gather(ids)
    torch.xpu.synchronize()
    eq = torch.equal(x.view(torch.int16), y.view(torch.int16))
    res.update(status="pass" if eq else "mismatch", shard_edges_checked=len(edges),
               differing_rows=int((x.view(torch.int16) != y.view(torch.int16)).any(-1).sum()),
               subset_rows=sub.num_rows, full_rows=full.num_rows, elapsed_s=round(time.time() - res["started"], 1))
    full.release(); del full
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(res))
    sys.exit(0 if eq else 1)
except Exception as e:
    res.update(status="error", error=f"{type(e).__name__}: {e}"[:1000], traceback=traceback.format_exc()[-3000:])
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("status", "error")}))
    sys.exit(3)
