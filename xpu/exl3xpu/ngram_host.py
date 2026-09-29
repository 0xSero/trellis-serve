"""EXL3 n-gram ("PLE") table of Qwen3.8-Flash-Next served from USM host memory on Intel XPU (B70).
Port of the 3090 integration lane's sglang_exl3/offload/ngram_host.py: the packed rows live in one
sycl::malloc_host allocation (exl3xpu_moe::host_alloc) and exl3xpu_moe::ngram_gather_dequant reads them zero-copy.

`ngram_embedding.safetensors` (format `exl3_ngram_trellis`) holds `<key>.shard_N.trellis [rows, 1 + 160*K/16] int16`
(row = fp16 scale word + a 160 x K bit tail-biting mul1 trellis), plus `head_bias [16,160]`, `head_offsets`,
`head_vocab_sizes` and `layer_multipliers`. SGLang's qwen4_exp wants a bf16/fp8 `VocabParallelEmbedding` of
320M x 160 (102 GB bf16) instead; `install()` swaps that embedding for `Exl3NgramHostTable`:

  * the packed rows (32.6 GB at K=5) are read once into anonymous host memory, page-locked and mapped
    (here: one sycl::malloc_host allocation);
  * forward(ids) takes the row ids SGLang already hashed on the GPU ([T, 16], head-major columns) and runs one kernel
    (csrc/offload/offload_kernels.cu: ngram_gather_dequant) that reads each 102-byte row zero-copy over PCIe and decodes
    it (+ per-head bias) -- no host involvement, so decode stays inside the CUDA graph.

The hash constants SGLang derives from the config are checked against the file's own tensors at load.
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import torch

logger = logging.getLogger(__name__)
FILE = "ngram_embedding.safetensors"
_TABLES: dict = {}
_DT = {"I16": torch.int16, "I64": torch.int64, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _read_small(path, base, info):
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        buf = bytearray(f.read(b - a))
    return torch.frombuffer(buf, dtype=_DT[info["dtype"]]).reshape(info["shape"]).clone()


class Exl3NgramHostTable(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        path = os.path.join(model_path, FILE)
        hdr, base = _header(path)
        meta = hdr.pop("__metadata__", {}) or {}
        if meta.get("format") != "exl3_ngram_trellis":
            raise ValueError(f"{path}: not an exl3_ngram_trellis table ({meta.get('format')!r})")
        shards = {}
        prefix = None
        for k, v in hdr.items():
            m = re.match(r"(.*)\.shard_(\d+)\.trellis$", k)
            if m:
                shards[int(m.group(2))] = v
                prefix = m.group(1)
        if not shards or sorted(shards) != list(range(len(shards))):
            raise ValueError(f"{path}: shard_N.trellis tensors missing or not contiguous")
        words = shards[0]["shape"][1]
        self.K = (words - 1) * 16 // 160
        if self.K != int(meta.get("K", self.K)) or 1 + 160 * self.K // 16 != words:
            raise ValueError(f"{path}: row width {words} does not match K={meta.get('K')}")
        self.words = words
        self.prefix = prefix
        rows_per = [shards[i]["shape"][0] for i in range(len(shards))]
        self.num_rows = sum(rows_per)
        row_bytes = words * 2
        nbytes = self.num_rows * row_bytes

        t0 = time.time()
        from .moe_offload import ops, s64
        X = ops()
        # host USM in <= 8 GiB chunks (one 32.6 GB sycl::malloc_host fails on this driver; 16 GiB works)
        self.rows_per_chunk = max(1, (8 << 30) // row_bytes)
        self._chunks = []
        for r in range(0, self.num_rows, self.rows_per_chunk):
            n = min(self.rows_per_chunk, self.num_rows - r)
            self._chunks.append(X.host_alloc(n * row_bytes))
        self.host = None
        # copy every shard to its row range: 64 MiB preads on a thread pool
        jobs, r0 = [], 0
        for i in range(len(shards)):
            a, b = shards[i]["data_offsets"]
            assert b - a == rows_per[i] * row_bytes
            off = r0 * row_bytes
            for c in range(0, b - a, 64 << 20):
                ln = min(64 << 20, b - a - c)
                jobs.append((base + a + c, off + c, ln))
            r0 += rows_per[i]
        mvs = [memoryview(c.numpy()) for c in self._chunks]
        cb = self.rows_per_chunk * row_bytes

        def dst_slices(dst, ln):
            """split a [dst, dst+ln) range of the logical table over the chunks"""
            out = []
            while ln > 0:
                ci, co = divmod(dst, cb)
                n = min(ln, cb - co)
                out.append((mvs[ci], co, n))
                dst += n; ln -= n
            return out
        fd = os.open(path, os.O_RDONLY)
        try:
            def rd(job):
                src_, dst, ln = job
                for mv, co, n_ in dst_slices(dst, ln):
                    got = 0
                    while got < n_:
                        n = os.preadv(fd, [mv[co + got: co + n_]], src_ + got)
                        if n <= 0:
                            raise IOError(f"{path}: short read at {src_ + got}")
                        got += n
                    src_ += n_
                return ln
            with ThreadPoolExecutor(int(os.environ.get("EXL3_NGRAM_LOAD_THREADS", "16"))) as ex:
                total = sum(ex.map(rd, jobs))
        finally:
            os.close(fd)
        del mvs
        t1 = time.time()
        self._X = X
        logger.info("EXL3 n-gram table: %d rows x %d words (K=%d) = %.2f GB in USM host memory (read %.1f s = %.1f GB/s)",
                    self.num_rows, words, self.K, nbytes / 1e9, t1 - t0, total / 1e9 / max(t1 - t0, 1e-6))

        dev = torch.device("xpu", torch.xpu.current_device())
        aux = {n: _read_small(path, base, hdr[f"{prefix}.{n}"]) for n in
               ("head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers") if f"{prefix}.{n}" in hdr}
        self.register_buffer("head_bias", aux["head_bias"].to(dev, torch.float16).contiguous(), persistent=False)
        self.register_buffer("chunk_ptrs", torch.tensor([s64(c.data_ptr()) for c in self._chunks], dtype=torch.int64,
                                                        device=dev), persistent=False)
        self.file_hash = {k: aux[k].long() for k in ("head_offsets", "head_vocab_sizes", "layer_multipliers")}
        self.num_heads = self.head_bias.shape[0]
        # SGLang reads these off its embedding modules
        self.embedding_dim = 160
        self.num_embeddings = self.num_rows
        self.quant_method = None
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.bfloat16, device=dev), persistent=False)

    def release(self) -> None:
        self._chunks = []

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # Qwen4ExpPinnedHostEmbedding interface (SGLang's PLE prefetch path, config.ple_offload_embedding): the gather runs
    # on the PLE prefetch stream while the preceding decoder layer computes
    def allocate_output(self, shape, device) -> torch.Tensor:
        with torch.inference_mode(False):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(self, ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if ids.shape[-1] != self.num_heads:
            raise ValueError(f"n-gram lookup expects [..., {self.num_heads}] ids, got {tuple(ids.shape)}")
        flat = ids.reshape(-1).to(torch.long).contiguous()
        if out is None:
            out = torch.empty((*ids.shape, 160), dtype=torch.bfloat16, device=ids.device)
        if out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() != flat.numel() * 160:
            raise ValueError(f"n-gram gather output {tuple(out.shape)} {out.dtype} does not fit {tuple(ids.shape)} ids")
        self._X.ngram_gather_dequant(flat, self.chunk_ptrs, self.rows_per_chunk, self.num_rows, self.words, self.K,
                                     self.head_bias, out)
        return out

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gather(ids).view(*ids.shape, 160)


def get_table(model_path: str) -> Exl3NgramHostTable:
    t = _TABLES.get(model_path)
    if t is None:
        if os.environ.get("EXL3_NGRAM_TIER", "host") == "nvme":
            from .ngram_nvme import Exl3NgramNvmeTable      # tier 2: NVMe rows + RAM row cache (X008)
            t = _TABLES[model_path] = Exl3NgramNvmeTable(model_path)
        else:
            t = _TABLES[model_path] = Exl3NgramHostTable(model_path)
    return t


def has_table(model_path) -> bool:
    return bool(model_path) and os.path.exists(os.path.join(model_path, FILE))


def install() -> None:
    """Patch sglang.srt.models.qwen4_exp.Qwen4ExpNGramEmbedding so an EXL3 checkpoint with an exl3_ngram_trellis table
    gets an Exl3NgramHostTable instead of the 102 GB bf16 VocabParallelEmbedding."""
    from sglang.srt.models import qwen4_exp as q
    cls = q.Qwen4ExpNGramEmbedding
    if getattr(cls, "_exl3_patched", False):
        return
    orig = cls.__init__

    def __init__(self, config, embedding_dim, ple_layer_index=0, quant_config=None, prefix=""):
        path = getattr(quant_config, "model_path", None) if getattr(quant_config, "get_name", lambda: "")() == "exl3" else None
        if not has_table(path):
            return orig(self, config, embedding_dim, ple_layer_index=ple_layer_index, quant_config=quant_config,
                        prefix=prefix)
        table = get_table(path)
        vpe = q.VocabParallelEmbedding

        def _factory(num_embeddings, embedding_dim_, *a, **k):
            if num_embeddings != table.num_rows or embedding_dim_ != 160:
                raise ValueError(f"n-gram table shape mismatch: SGLang wants {num_embeddings} x {embedding_dim_}, "
                                 f"file has {table.num_rows} x 160")
            return table
        q.VocabParallelEmbedding = _factory
        try:
            orig(self, config, embedding_dim, ple_layer_index=ple_layer_index, quant_config=quant_config, prefix=prefix)
        finally:
            q.VocabParallelEmbedding = vpe
        mine = {"head_offsets": self.ngram_heads_offsets, "head_vocab_sizes": self.ngram_heads_vocab_sizes,
                "layer_multipliers": self.layer_multipliers}
        for name, buf in mine.items():
            ref = table.file_hash[name]
            if tuple(buf.shape) != tuple(ref.shape) or not torch.equal(buf.cpu(), ref):
                logger.warning("n-gram hash constant %s differs from the file (%s vs %s): using the file's", name,
                               buf.cpu().tolist()[:4], ref.tolist()[:4])
                buf.copy_(ref.to(buf.device))
        logger.info("%s: n-gram table served from %s (EXL3 K=%d rows, gather + decode on the GPU)", prefix,
                    "NVMe + RAM row cache (tier 2)" if type(table).__name__ == "Exl3NgramNvmeTable"
                    else "USM host memory (zero-copy)", table.K)

    cls.__init__ = __init__
    cls._exl3_patched = True

    # With config.ple_offload_embedding SGLang wraps the table in Qwen4ExpPinnedHostEmbedding (bf16/fp8 host copy);
    # ours already lives in host memory and implements the same gather/allocate_output/reduce interface: pass through.
    base = q.Qwen4ExpPinnedHostEmbedding

    class Qwen4ExpPinnedHostEmbeddingExl3(base):
        def __new__(klass, embedding, *a, **k):
            if isinstance(embedding, Exl3NgramHostTable):
                return embedding
            obj = base.__new__(base)
            obj.__init__(embedding, *a, **k)
            return obj
    q.Qwen4ExpPinnedHostEmbedding = Qwen4ExpPinnedHostEmbeddingExl3
