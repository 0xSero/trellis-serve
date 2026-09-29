"""K02 trace statistics: per-layer popularity skew (share of decode accesses covered by the top-N experts of that
layer, over all requests), step-to-step reuse (fraction of step t's experts also routed at step t-1 / within the
last 8 / 64 steps), distinct experts per layer per request."""
import glob, json, os, sys
import numpy as np
d = sys.argv[1]
L, E, K = 48, 512, 10
reqs = {os.path.basename(f)[:-4]: np.load(f)["decode_ids"].astype(np.int64) for f in sorted(glob.glob(os.path.join(d, "*.npz")))}
allids = np.concatenate(list(reqs.values()))                       # [S, L, K]
cnt = np.zeros((L, E), np.int64)
for l in range(L):
    cnt[l] = np.bincount(allids[:, l].reshape(-1), minlength=E)
srt = -np.sort(-cnt, axis=1)
cov = lambda n: (srt[:, :n].sum(1) / srt.sum(1))
out = {"top64_share_mean": float(cov(64).mean()), "top128_share_mean": float(cov(128).mean()),
       "top195_share_mean": float(cov(195).mean()), "top256_share_mean": float(cov(256).mean()),
       "top195_share_by_layer": [round(float(v), 3) for v in cov(195)],
       "never_routed_in_decode_per_layer_mean": float((cnt == 0).sum(1).mean())}
reuse = {1: [], 8: [], 64: []}
for ids in reqs.values():
    S = ids.shape[0]
    for w in reuse:
        hits = tot = 0
        for l in range(0, L, 1):
            seq = ids[:, l]
            last = np.full(E, -10**9)
            for t in range(S):
                row = seq[t]
                hits += int(((t - last[row]) <= w).sum()); tot += K
                last[row] = t
        reuse[w].append(hits / tot)
out["reuse_within_1_8_64_steps"] = {w: round(float(np.mean(v)), 4) for w, v in reuse.items()}
out["distinct_per_layer_per_request_mean"] = {k: round(float(np.mean([len(np.unique(v[:, l])) for l in range(L)])), 1) for k, v in reqs.items()}
print(json.dumps(out, indent=1))
