"""K07 miss-path study on the K02 traces: can layer l+1's misses be prefetched while layer l runs?
Global LRU cache of B slots (K03), requests in K03 order; train predictors on the first half of the requests (by
order), evaluate on the second half (cache warm from the first half). Before layer l+1 of step t is accessed, up to K
predicted experts of layer l+1 that are NOT resident are inserted (prefetched) as MRU. Predictors:
  cooc : sum of co-occurrence counts C[l][e, e'] (e routed at layer l this step, e' routed at l+1 in the same token)
  prev : the previous token's layer-(l+1) experts
Metric: hit rate of the real accesses, prefetch precision (prefetched experts used at that layer/step), PCIe bytes."""
import glob, os, sys, json
from collections import OrderedDict
import numpy as np

L, E, K = 48, 512, 10
d = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
files = sorted(glob.glob(os.path.join(d, "*.npz")))
reqs = [np.load(f)["decode_ids"].astype(np.int64) for f in files]
order = np.random.default_rng(0).permutation(len(reqs))
reqs = [reqs[i] for i in order]
half = len(reqs) // 2
C = np.zeros((L - 1, E, E), dtype=np.int32)
for r in reqs[:half]:
    for l in range(L - 1):
        a, b = r[:, l], r[:, l + 1]                                   # [S, K]
        np.add.at(C[l], (np.repeat(a, K, axis=1).ravel(), np.tile(b, (1, K)).ravel()), 1)


def run(pred, kpf):
    lru = OrderedDict()
    def access(key, admit=True):
        if key in lru:
            lru.move_to_end(key); return True
        if admit:
            if len(lru) >= B:
                lru.popitem(last=False)
            lru[key] = None
        return False
    hits = tot = pf = pf_used = 0
    for ri, r in enumerate(reqs):
        test = ri >= half
        prev = None
        for t in range(r.shape[0]):
            pending = set()
            for l in range(L):
                row = r[t, l]
                for e in row:
                    h = access(l * E + int(e))
                    if test:
                        hits += h; tot += 1
                        if int(e) in pending:
                            pf_used += 1
                if kpf and l + 1 < L:
                    if pred == "cooc":
                        sc = C[l][row].sum(0)
                        cand = np.argsort(-sc)[:4 * kpf]
                    else:
                        cand = r[t - 1, l + 1] if t > 0 else []
                    pending, n = set(), 0
                    for e2 in cand:
                        key = (l + 1) * E + int(e2)
                        if key not in lru:
                            access(key); pending.add(int(e2)); n += 1
                            if test:
                                pf += 1
                            if n >= kpf:
                                break
                else:
                    pending = set()
    return {"pred": pred, "K": kpf, "hit_rate": hits / tot, "prefetch_per_token": pf / max(1, tot / (L * K)) ,
            "prefetch_precision": pf_used / max(1, pf)}


out = [run("none", 0)]
print(json.dumps(out[-1]), flush=True)
for pred in ("cooc", "prev"):
    for kpf in (1, 2, 3, 5):
        out.append(run(pred, kpf)); print(json.dumps(out[-1]), flush=True)
json.dump(out, open(f"/Users/sero/freetoken-exl3/runs/2026-09-29-K03-sim/prefetch_sim_{B}.json", "w"), indent=1)
