"""Pinned-RAM reference over rows [0, N) of the n-gram table, shared by test_ngram_nvme_stress.py and
validate_subset_ref.py (which checks it against the full pinned table)."""
import os, re
import torch
from exl3xpu.ngram_host import FILE, _header, _read_small


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
