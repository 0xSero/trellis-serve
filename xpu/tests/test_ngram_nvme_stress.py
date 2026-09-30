"""Fault-exposure and integrity stress for the NVMe n-gram tier's GPU<->host handshake (nvme_publish / nvme_wait).

The arm is chosen by EXL3_NVME_PUBLISH (store | atomic, read by the extension) and EXL3_NVME_LIB (which .so). Loops until
--seconds elapse:
  eager lookups of mixed sizes (decode-sized 16 ids up to the 524,288-id capacity), back to back with no host sync
  between some of them (rapid buffer reuse), and
  XPU-graph replays of decode-shaped lookups (T=1, T=2) with fresh ids copied in before every replay, as SGLang decode does.
--seconds is stress-loop time, measured after setup (exposure_s). Every result is compared bit for bit with a pinned-RAM
reference (the pinned tier's own gather/dequant kernel over one host chunk: a separate, atomic-free path). The reference
holds only rows [0, --ref-rows) and every test id is drawn from that range, so the check costs ~2 GB of pinned RAM
instead of the full 32.6 GB table (which left the dev window ~7 GB above its memory floor). The publish/wait handshake
under test writes the same buffers whatever the id values are. The extension's error word and the GPU wait count are
checked against the number of lookups issued. Progress JSON is rewritten every 10 s so the exposure before a device
loss survives the process.
Exit: 0 clean, 1 integrity failure, 3 device/runtime error.
"""
import argparse, json, os, re, sys, time, traceback
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exl3xpu.ngram_host import FILE, _header, _read_small
from exl3xpu.ngram_nvme import Exl3NgramNvmeTable

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/model")
ap.add_argument("--seconds", type=float, default=600)
ap.add_argument("--ram-gb", type=float, default=0.5, help="NVMe tier RAM row cache (small, so lookups still miss)")
ap.add_argument("--ref-rows", type=int, default=20_000_000, help="rows held by the reference; ids are drawn below this")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()


class SubsetRef:
    """Pinned-RAM reference over rows [0, rows): Exl3NgramHostTable's gather (ngram_gather_dequant) with one chunk."""

    def __init__(self, model, rows):
        from exl3xpu.moe_offload import ops, s64
        path = os.path.join(model, FILE)
        hdr, base = _header(path)
        meta = hdr.pop("__metadata__", {}) or {}
        if meta.get("format") != "exl3_ngram_trellis":
            raise ValueError(f"{path}: not an exl3_ngram_trellis table ({meta.get('format')!r})")
        shards, prefix = {}, None
        for k, v in hdr.items():
            m = re.match(r"(.*)\.shard_(\d+)\.trellis$", k)
            if m:
                shards[int(m.group(2))] = v
                prefix = m.group(1)
        if not shards or sorted(shards) != list(range(len(shards))):
            raise ValueError(f"{path}: shard_N.trellis tensors missing or not contiguous")
        self.words = shards[0]["shape"][1]
        self.K = (self.words - 1) * 16 // 160
        if self.K != int(meta.get("K", self.K)) or 1 + 160 * self.K // 16 != self.words:
            raise ValueError(f"{path}: row width {self.words} does not match K={meta.get('K')}")
        if any(shards[i]["shape"][1] != self.words for i in shards):
            raise ValueError(f"{path}: shards differ in row width")
        rb = self.words * 2
        self.num_rows = min(rows, sum(shards[i]["shape"][0] for i in shards))
        self.X = ops()
        self.chunk = self.X.host_alloc(self.num_rows * rb)
        mv = memoryview(self.chunk.numpy())
        fd = os.open(path, os.O_RDONLY)
        try:
            done = 0
            for i in range(len(shards)):
                if done >= self.num_rows:
                    break
                a0, _ = shards[i]["data_offsets"]
                ln = min(shards[i]["shape"][0], self.num_rows - done) * rb
                off = 0
                while off < ln:
                    n = os.preadv(fd, [mv[done * rb + off: done * rb + min(ln, off + (64 << 20))]], base + a0 + off)
                    if n <= 0:
                        raise IOError(f"{path}: short read")
                    off += n
                done += ln // rb
        finally:
            os.close(fd)
        d = torch.device("xpu", torch.xpu.current_device())
        self.head_bias = _read_small(path, base, hdr[f"{prefix}.head_bias"]).to(d, torch.float16).contiguous()
        self.chunk_ptrs = torch.tensor([s64(self.chunk.data_ptr())], dtype=torch.int64, device=d)

    def gather(self, ids):
        flat = ids.reshape(-1).to(torch.long).contiguous()
        out = torch.empty((*ids.shape, 160), dtype=torch.bfloat16, device=ids.device)
        self.X.ngram_gather_dequant(flat, self.chunk_ptrs, self.num_rows, self.num_rows, self.words, self.K,
                                    self.head_bias, out)
        return out

dev = torch.device("xpu", 0)
g = torch.Generator().manual_seed(a.seed)
st = dict(arm=os.environ.get("EXL3_NVME_PUBLISH", "store"), lib=os.environ.get("EXL3_NVME_LIB", "default"),
          seed=a.seed, seconds_budget=a.seconds, started=time.time(), eager_lookups=0, graph_replays=0, ids=0,
          mismatches=0, max_ids=0, status="setup", error=None)


def dump():
    st["elapsed_s"] = round(time.time() - st["started"], 1)
    # exposure counts only the stress loop, not table setup or graph capture
    st["exposure_s"] = round(time.time() - st["run_started"], 1) if st.get("run_started") else 0.0
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, a.out)


def rand_ids(T, hot):
    # half the rows from a small hot set (cache hits), half uniform (misses -> NVMe reads)
    ids = torch.randint(0, ho.num_rows, (T, 16), generator=g, dtype=torch.int64)
    m = torch.rand((T, 16), generator=g) < 0.5
    ids[m] = hot[torch.randint(0, hot.numel(), (int(m.sum()),), generator=g)]
    return ids.to(dev)


def check(out, ids):
    ref = ho.gather(ids)
    eq = torch.equal(out.view(torch.int16), ref.view(torch.int16))
    st["mismatches"] += not eq
    return eq


try:
    dump()
    t0 = time.time()
    nv = Exl3NgramNvmeTable(a.model, ram_gb=a.ram_gb, device=dev)
    ho = SubsetRef(a.model, a.ref_rows)
    st["ref_rows"] = ho.num_rows; st["ram_gb"] = a.ram_gb
    st["setup_s"] = round(time.time() - t0, 1)
    hot = torch.randint(0, ho.num_rows, (4096,), generator=g, dtype=torch.int64)
    # decode graphs, captured as SGLang does (side stream warm-up, then capture)
    graphs = {}
    for T in (1, 2):
        ids_s = rand_ids(T, hot)
        out_s = torch.empty((T, 16, 160), dtype=torch.bfloat16, device=dev)
        s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            nv.gather(ids_s, out=out_s)
        torch.xpu.current_stream().wait_stream(s); torch.xpu.synchronize()
        gr = torch.xpu.XPUGraph()
        with torch.xpu.graph(gr):
            nv.gather(ids_s, out=out_s)
        graphs[T] = (gr, ids_s, out_s)
    st["status"] = "running"; st["run_started"] = time.time(); dump()
    sizes = [1] * 8 + [2] * 4 + [7, 64, 512, 2048, 4096, 16384, 32768]
    last = time.time()
    while time.time() - st["run_started"] < a.seconds:
        # eager burst: 8 lookups issued back to back (no sync), then all checked
        burst = []
        for _ in range(8):
            T = sizes[int(torch.randint(0, len(sizes), (1,), generator=g))]
            ids = rand_ids(T, hot)
            burst.append((ids, nv.gather(ids)))
            st["eager_lookups"] += 1; st["ids"] += T * 16; st["max_ids"] = max(st["max_ids"], T * 16)
        torch.xpu.synchronize()
        for ids, out in burst:
            check(out, ids)
        # graph burst: 200 decode replays, alternating T=1 / T=2, fresh ids each time, checked every replay
        for k in range(200):
            gr, ids_s, out_s = graphs[1 + (k & 1)]
            ids_s.copy_(rand_ids(ids_s.shape[0], hot))
            gr.replay()
            torch.xpu.synchronize()
            check(out_s, ids_s)
            st["graph_replays"] += 1; st["ids"] += ids_s.numel()
        if time.time() - last > 10:
            s_ = nv.stats()
            st["ext"] = {k: s_[k] for k in ("lookups", "misses", "error", "gpu_waits", "gpu_wait_spins_max", "bytes")}
            dump(); last = time.time()
    torch.xpu.synchronize()
    s_ = nv.stats()
    st["ext"] = {k: s_[k] for k in ("lookups", "misses", "error", "gpu_waits", "gpu_wait_spins_max", "bytes")}
    # every lookup: 1 capture warm-up per graph + 1 capture (not executed) + eager + replays
    st["expected_gpu_waits_min"] = st["eager_lookups"] + st["graph_replays"]
    ok = st["mismatches"] == 0 and s_["error"] == 0 and s_["gpu_waits"] >= st["expected_gpu_waits_min"]
    st["status"] = "pass" if ok else "integrity_fail"
    dump()
    nv.release()
    sys.exit(0 if ok else 1)
except Exception as e:  # device loss surfaces here (UR_RESULT_ERROR_DEVICE_LOST)
    st["status"] = "runtime_error"
    st["error"] = f"{type(e).__name__}: {e}"[:2000]
    st["traceback"] = traceback.format_exc()[-4000:]
    dump()
    sys.exit(3)
