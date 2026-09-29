"""EXL3 n-gram ("PLE") table of Qwen3.8-Flash-Next served from pinned host memory.

`ngram_embedding.safetensors` (format `exl3_ngram_trellis`) holds `<key>.shard_N.trellis [rows, 1 + 160*K/16] int16`
(row = fp16 scale word + a 160 x K bit tail-biting mul1 trellis), plus `head_bias [16,160]`, `head_offsets`,
`head_vocab_sizes` and `layer_multipliers`. SGLang's qwen4_exp wants a bf16/fp8 `VocabParallelEmbedding` of
320M x 160 (102 GB bf16) instead; `install()` swaps that embedding for `Exl3NgramHostTable`:

  * the packed rows (32.6 GB at K=5) are read once into anonymous host memory, page-locked and mapped
    (cudaHostRegister PORTABLE|MAPPED);
  * forward(ids) takes the row ids SGLang already hashed on the GPU ([T, 16], head-major columns) and runs one kernel
    (csrc/offload/offload_kernels.cu: ngram_gather_dequant) that reads each 102-byte row zero-copy over PCIe and decodes
    it (+ per-head bias) -- no host involvement, so decode stays inside the CUDA graph.

The hash constants SGLang derives from the config are checked against the file's own tensors at load.
"""
from __future__ import annotations

import json
import logging
import mmap
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
        from exllamav3.model.model_tp_cuda import (CUDA_HOST_REGISTER_MAPPED, CUDA_HOST_REGISTER_PORTABLE,
                                                   cuda_host_get_device_pointer, cuda_host_register)
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
        self._map = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        try:
            self._map.madvise(mmap.MADV_HUGEPAGE)
        except Exception:
            pass
        self.host = torch.frombuffer(self._map, dtype=torch.int16, count=nbytes // 2).view(self.num_rows, words)
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
        mv = memoryview(self._map)
        fd = os.open(path, os.O_RDONLY)
        try:
            def rd(job):
                src, dst, ln = job
                got = 0
                while got < ln:
                    n = os.preadv(fd, [mv[dst + got: dst + ln]], src + got)
                    if n <= 0:
                        raise IOError(f"{path}: short read at {src + got}")
                    got += n
                return ln
            with ThreadPoolExecutor(int(os.environ.get("SGLANG_EXL3_NGRAM_LOAD_THREADS", "16"))) as ex:
                total = sum(ex.map(rd, jobs))
        finally:
            os.close(fd)
        del mv
        t1 = time.time()
        ptr = self.host.data_ptr()
        cuda_host_register(ptr, nbytes, flags=CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
        self.dev_ptr = cuda_host_get_device_pointer(ptr)
        t2 = time.time()
        logger.info("EXL3 n-gram table: %d rows x %d words (K=%d) = %.2f GB in pinned host memory (read %.1f s = %.1f GB/s, "
                    "register %.1f s)", self.num_rows, words, self.K, nbytes / 1e9, t1 - t0, total / 1e9 / max(t1 - t0, 1e-6),
                    t2 - t1)

        dev = torch.device("cuda", torch.cuda.current_device())
        aux = {n: _read_small(path, base, hdr[f"{prefix}.{n}"]) for n in
               ("head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers") if f"{prefix}.{n}" in hdr}
        self.register_buffer("head_bias", aux["head_bias"].to(dev, torch.float16).contiguous(), persistent=False)
        self.file_hash = {k: aux[k].long() for k in ("head_offsets", "head_vocab_sizes", "layer_multipliers")}
        self.num_heads = self.head_bias.shape[0]
        from . import _ext
        self._ext = _ext.load()
        # SGLang reads these off its embedding modules
        self.embedding_dim = 160
        self.num_embeddings = self.num_rows

    def release(self) -> None:
        """Unregister and unmap the host table. Must precede freeing the mapping: a registration that outlives its
        mapping keeps the old pages behind that VA range in CUDA's UVA map, and a later mapping placed at the same
        address (e.g. the CPU MoE worker's shared control block) would be read/written by the GPU through the stale
        pages (seen as a lost-flag hang in offload_selftest)."""
        if getattr(self, "host", None) is None:
            return
        from exllamav3.model.model_tp_cuda import cuda_host_unregister
        torch.cuda.synchronize()
        cuda_host_unregister(self.host.data_ptr())
        self.host = None
        try:
            self._map.close()
        except BufferError:
            pass

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
        self._ext.ngram_gather_dequant(flat, self.dev_ptr, self.num_rows, self.words, self.K, self.head_bias, out)
        return out

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gather(ids).view(*ids.shape, 160)


def tier() -> str:
    """SGLANG_EXL3_NGRAM_TIER: `pinned` (default: whole table in pinned RAM) or `nvme` (ngram_nvme: rows read from the
    safetensors file with O_DIRECT behind a SGLANG_EXL3_NGRAM_RAM_GB row cache)."""
    t = os.environ.get("SGLANG_EXL3_NGRAM_TIER", "pinned").strip().lower() or "pinned"
    if t not in ("pinned", "nvme"):
        raise ValueError(f"SGLANG_EXL3_NGRAM_TIER={t!r}: expected pinned or nvme")
    return t


def get_table(model_path: str):
    t = _TABLES.get(model_path)
    if t is None:
        if tier() == "nvme":
            from .ngram_nvme import Exl3NgramNvmeTable
            t = Exl3NgramNvmeTable(model_path)
        else:
            t = Exl3NgramHostTable(model_path)
        _TABLES[model_path] = t
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
        from ..sglang_glue.config import Exl3Config
        path = getattr(quant_config, "model_path", None) if isinstance(quant_config, Exl3Config) else None
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
        table.eos_token_id = int(getattr(self, "eos_token_id", -1))
        logger.info("%s: n-gram table served from %s (EXL3 K=%d rows, zero-copy gather + decode)", prefix,
                    "NVMe + RAM row cache" if tier() == "nvme" else "pinned host memory", table.K)

    cls.__init__ = __init__
    cls._exl3_patched = True
    if tier() == "nvme":
        from .ngram_nvme import install_prefill_hints
        install_prefill_hints()

    # With config.ple_offload_embedding SGLang wraps the table in Qwen4ExpPinnedHostEmbedding (bf16/fp8 host copy);
    # ours already lives in host memory and implements the same gather/allocate_output/reduce interface: pass through.
    base = q.Qwen4ExpPinnedHostEmbedding

    class Qwen4ExpPinnedHostEmbeddingExl3(base):
        def __new__(klass, embedding, *a, **k):
            if isinstance(embedding, Exl3NgramHostTable) or type(embedding).__name__ == "Exl3NgramNvmeTable":
                return embedding
            obj = base.__new__(base)
            obj.__init__(embedding, *a, **k)
            return obj
    q.Qwen4ExpPinnedHostEmbedding = Qwen4ExpPinnedHostEmbeddingExl3
