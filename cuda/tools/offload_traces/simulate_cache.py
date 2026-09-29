"""K03: expert-cache policy simulation on real routing traces (K02 npz files).

Access stream: requests in a fixed shuffled order (seed 0), each request's decode steps in order, per step layers 0..47,
per layer its 10 routed experts (key = layer * 512 + expert). The cache persists across requests (a server). A hit = a
routed (layer, expert) resident at access time. Policies (global pool of B slots unless noted):
  lru          global LRU, admit every miss
  lru_layer    per-layer LRU, B // 48 slots per layer
  lfu          global LFU (cumulative decode counts, LRU tie-break), admit every miss
  arc          ARC (Megiddo & Modha), global
  clock        CLOCK second chance (what om.ExpertCache implements), with in-step pinning
  static_prof  static hot set = top-B keys by frequency over ALL requests' PREFILL routing (profile), no admission
  static_own   per request: top-B keys of that request's own prefill routing, no admission during decode
  lru_seeded   static_own at request start, then global LRU with admission
  opt          Belady (evict the key used furthest in the future): upper bound
Costs per layer and step (m misses of the 10 routed experts, h = 10 - m hits):
  gpu    : 5.5 h + 73 m us                         (device hit, zero-copy miss incl. fused admission; K01)
  hybrid : min over F in [0, m] of max(5.5 h + 73 F, 65 (m - F)) us  (F misses zero-copy on the GPU, the rest on the
           CPU in parallel at 65 us/expert = exllamav3 CPU worker, 480 experts in ~31 ms); for the hybrid the hit rate
           is re-simulated with only the GPU-served misses admitted (hybrid_adm) and with all admitted (upper bound)
Outputs a JSON with hit rates (overall, per category), MoE ms/token (gpu, hybrid) per policy and budget.
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

L, E, K = 48, 512, 10
HIT_US, MISS_US, CPU_US = 5.5, 73.0, 65.0
CPU_FIXED_US = 0.0     # per-layer CPU job overhead when C > 0 (issue/collect handshake); set from K04 via --cpu-fixed-us


def load(trace_dir):
    reqs = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.npz"))):
        z = np.load(f)
        reqs.append({"name": os.path.basename(f)[:-4], "category": str(z["category"]),
                     "decode": z["decode_ids"].astype(np.int32), "prefill": z["prefill_ids"].astype(np.int32)})
    rng = np.random.default_rng(0)
    order = rng.permutation(len(reqs))
    return [reqs[i] for i in order]


def hybrid_layer_cost(m):
    """min_F max(HIT*h + MISS*F, CPU*(m-F)) for every m in 0..10 (h = 10 - m) -> (cost us, F)."""
    out = []
    for mm in range(K + 1):
        h = K - mm
        best = min((max(HIT_US * h + MISS_US * f, CPU_US * (mm - f) + (CPU_FIXED_US if mm - f > 0 else 0.0)), f)
                   for f in range(mm + 1))
        out.append(best)
    return out


HYB = hybrid_layer_cost(0)


class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def access(self, key, admit=True):
        if key in self.d:
            self.d.move_to_end(key)
            return True
        if admit and self.cap > 0:
            if len(self.d) >= self.cap:
                self.d.popitem(last=False)
            self.d[key] = None
        return False

    def seed(self, keys):
        self.d = OrderedDict((k, None) for k in keys[: self.cap])


class LRULayer:
    def __init__(self, cap):
        self.per = [LRU(cap // L) for _ in range(L)]

    def access(self, key, admit=True):
        return self.per[key // E].access(key, admit)


class LFU:
    def __init__(self, cap):
        self.cap, self.cnt, self.res, self.heap, self.t = cap, defaultdict(int), {}, [], 0

    def access(self, key, admit=True):
        self.t += 1
        self.cnt[key] += 1
        if key in self.res:
            self.res[key] = self.t
            heapq.heappush(self.heap, (self.cnt[key], self.t, key))
            return True
        if admit and self.cap > 0:
            while len(self.res) >= self.cap:
                c, t, k = heapq.heappop(self.heap)
                if k in self.res and self.res[k] == t and self.cnt[k] == c:
                    del self.res[k]
            self.res[key] = self.t
            heapq.heappush(self.heap, (self.cnt[key], self.t, key))
        return False


class ARC:
    def __init__(self, cap):
        self.c, self.p = cap, 0
        self.t1, self.t2, self.b1, self.b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    def _replace(self, key):
        if self.t1 and (len(self.t1) > self.p or (key in self.b2 and len(self.t1) == self.p)):
            k, _ = self.t1.popitem(last=False); self.b1[k] = None
        else:
            k, _ = self.t2.popitem(last=False); self.b2[k] = None

    def access(self, key, admit=True):
        c = self.c
        if key in self.t1:
            del self.t1[key]; self.t2[key] = None; return True
        if key in self.t2:
            self.t2.move_to_end(key); return True
        if not admit or c == 0:
            return False
        if key in self.b1:
            self.p = min(c, self.p + max(len(self.b2) // max(len(self.b1), 1), 1))
            self._replace(key); del self.b1[key]; self.t2[key] = None; return False
        if key in self.b2:
            self.p = max(0, self.p - max(len(self.b1) // max(len(self.b2), 1), 1))
            self._replace(key); del self.b2[key]; self.t2[key] = None; return False
        if len(self.t1) + len(self.b1) == c:
            if len(self.t1) < c:
                self.b1.popitem(last=False); self._replace(key)
            else:
                self.t1.popitem(last=False)
        elif len(self.t1) + len(self.b1) < c and len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
            if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) == 2 * c:
                self.b2.popitem(last=False)
            self._replace(key)
        self.t1[key] = None
        return False


class Clock:
    """CLOCK with in-step pinning, as csrc/exl3_offload_cache.cu (stamp == current step never evicted)."""
    def __init__(self, cap):
        self.cap = cap
        self.owner = [-1] * cap
        self.ref = [0] * cap
        self.stamp = [-1] * cap
        self.slot = {}
        self.hand = 0
        self.step = 0

    def new_step(self):
        self.step += 1

    def access(self, key, admit=True):
        s = self.slot.get(key)
        if s is not None:
            self.stamp[s] = self.step; self.ref[s] = 1
            return True
        if not admit or self.cap == 0:
            return False
        for _ in range(2 * self.cap + 1):
            v = self.hand
            self.hand = (self.hand + 1) % self.cap
            if self.owner[v] < 0:
                break
            if self.stamp[v] == self.step:
                continue
            if self.ref[v]:
                self.ref[v] = 0; continue
            break
        else:
            return False
        o = self.owner[v]
        if o >= 0:
            del self.slot[o]
        self.owner[v], self.slot[key], self.stamp[v], self.ref[v] = key, v, self.step, 1
        return False


class Static:
    def __init__(self, keys):
        self.s = set(keys)

    def access(self, key, admit=True):
        return key in self.s


def topk_keys(ids, cap):
    """ids [N, L, K] (-1 = none) -> the cap most frequent keys."""
    lay = np.broadcast_to(np.arange(L)[None, :, None], ids.shape)
    keys = (lay * E + ids)[ids >= 0]
    cnt = np.bincount(keys, minlength=L * E)
    return [int(k) for k in np.argsort(-cnt, kind="stable")[:cap] if cnt[k] > 0]


def run_policy(reqs, policy, cap, profile_keys=None, hybrid_admit=False):
    """-> miss counts per (step, layer) for the whole stream, and per-request hit stats."""
    misses_all, per_req = [], []
    if policy == "lru" or policy == "lru_seeded":
        c = LRU(cap)
    elif policy == "lru_layer":
        c = LRULayer(cap)
    elif policy == "lfu":
        c = LFU(cap)
    elif policy == "arc":
        c = ARC(cap)
    elif policy == "clock":
        c = Clock(cap)
    elif policy == "static_prof":
        c = Static(profile_keys[:cap])
    elif policy == "static_own":
        c = None
    else:
        raise ValueError(policy)
    for r in reqs:
        d = r["decode"]
        if policy == "static_own":
            c = Static(topk_keys(r["prefill"], cap))
        elif policy == "lru_seeded":
            seed = topk_keys(r["prefill"], cap)
            keep = [k for k in c.d if k not in set(seed)]            # prefill-hot keys become MRU
            c.d = OrderedDict((k, None) for k in (keep + seed[::-1])[-cap:]) if cap else OrderedDict()
        miss = np.zeros((d.shape[0], L), dtype=np.int8)
        for t in range(d.shape[0]):
            if policy == "clock":
                c.new_step()
            for l in range(L):
                row = d[t, l]
                base = l * E
                m = 0
                if hybrid_admit:
                    # decide F from the hit count first (hits do not depend on the order within a layer)
                    keys = [base + int(e) for e in row]
                    res = [c.access(k, admit=False) for k in keys]
                    mm = res.count(False)
                    f = HYB[mm][1]
                    for k, hit in zip(keys, res):
                        if not hit:
                            if f > 0:
                                c.access(k, admit=True); f -= 1
                    m = mm
                else:
                    for e in row:
                        if not c.access(base + int(e)):
                            m += 1
                miss[t, l] = m
        misses_all.append(miss)
        per_req.append({"name": r["name"], "category": r["category"], "steps": int(d.shape[0]),
                        "hit_rate": 1.0 - float(miss.sum()) / (d.shape[0] * L * K)})
    return misses_all, per_req


def opt_misses(reqs, cap):
    """Belady over the concatenated decode stream (admit every miss, evict furthest next use)."""
    seq = np.concatenate([(np.arange(L)[None, :, None] * E + r["decode"]).reshape(-1) for r in reqs])
    n = seq.size
    nxt = np.empty(n, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        k = int(seq[i]); nxt[i] = last.get(k, n + i); last[k] = i
    res, heap, miss = {}, [], np.zeros(n, dtype=bool)
    for i in range(n):
        k = int(seq[i])
        if k in res:
            res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
            continue
        miss[i] = True
        if cap == 0:
            continue
        while len(res) >= cap:
            nn, kk = heapq.heappop(heap)
            if kk in res and res[kk] == -nn:
                del res[kk]
        res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
    out, pos = [], 0
    for r in reqs:
        s = r["decode"].shape[0] * L * K
        out.append(miss[pos:pos + s].reshape(-1, L, K).sum(-1).astype(np.int8)); pos += s
    return out


def costs(misses):
    m = np.concatenate(misses)                                  # [steps, L]
    h = K - m
    gpu = (HIT_US * h + MISS_US * m).sum(1)                     # us per token
    hyb_tab = np.array([c for c, _ in HYB])
    hyb = hyb_tab[m].sum(1)
    return float(gpu.mean() / 1e3), float(hyb.mean() / 1e3), float(1 - m.sum() / (m.size * K))


def main():
    global HYB, CPU_US, CPU_FIXED_US, HIT_US, MISS_US
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--budgets", default="4000,6000,8000,9400,12000")
    ap.add_argument("--policies", default="lru,lru_layer,lfu,arc,clock,static_prof,static_own,lru_seeded,opt")
    ap.add_argument("--hybrid-policies", default="lru,arc,clock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-us", type=float, default=65.0)
    ap.add_argument("--cpu-fixed-us", type=float, default=0.0)
    ap.add_argument("--hit-us", type=float, default=5.5)
    ap.add_argument("--miss-us", type=float, default=73.0)
    a = ap.parse_args()
    CPU_US, CPU_FIXED_US, HIT_US, MISS_US = a.cpu_us, a.cpu_fixed_us, a.hit_us, a.miss_us
    HYB = hybrid_layer_cost(0)
    reqs = load(a.trace_dir)
    prof = topk_keys(np.concatenate([r["prefill"] for r in reqs]), L * E)
    tot = sum(r["decode"].shape[0] for r in reqs)
    print(f"{len(reqs)} requests, {tot} decode steps, {tot * L * K} expert accesses", flush=True)
    res = {"requests": [(r["name"], r["category"], int(r["decode"].shape[0])) for r in reqs], "results": [],
           "cost_model_us": {"hit": HIT_US, "miss_zero_copy": MISS_US, "cpu": CPU_US, "cpu_fixed_per_layer": CPU_FIXED_US},
           "cpu_only_ms_per_token": CPU_US * L * K / 1e3, "gpu_zero_copy_only_ms": MISS_US * L * K / 1e3}
    for B in [int(b) for b in a.budgets.split(",")]:
        for pol in a.policies.split(","):
            mis = opt_misses(reqs, B) if pol == "opt" else run_policy(reqs, pol, B, prof)[0]
            per = None
            if pol != "opt":
                pass
            gpu_ms, hyb_ms, hr = costs(mis)
            cat = {}
            for r, mm in zip(reqs, mis):
                cat.setdefault(r["category"], []).append(mm)
            cat_hr = {k: round(1 - float(np.concatenate(v).sum()) / (np.concatenate(v).size * K), 4) for k, v in cat.items()}
            row = {"budget": B, "policy": pol, "hit_rate": round(hr, 4), "hit_rate_by_category": cat_hr,
                   "moe_ms_gpu": round(gpu_ms, 2), "moe_ms_hybrid_all_admitted": round(hyb_ms, 2)}
            if pol in a.hybrid_policies.split(","):
                mis_h = run_policy(reqs, pol, B, prof, hybrid_admit=True)[0]
                _, hyb2, hr2 = costs(mis_h)
                row.update({"hit_rate_hybrid_adm": round(hr2, 4), "moe_ms_hybrid": round(hyb2, 2)})
            res["results"].append(row)
            print(json.dumps(row), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
