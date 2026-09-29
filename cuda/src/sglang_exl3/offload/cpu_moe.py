"""Exl3CpuMoEMethod: routed EXL3 experts computed by exllamav3's persistent CPU MoE worker (tier 1 = host RAM).

The expert weights never reach the GPU: exllamav3's `MoeCpuHost` spawns one child process that reads every registered
layer's expert tensors straight from the checkpoint into its (hugepage) arena and runs the mul1 trellis GEMMs on the
CPU (AVX2 / AVX-512 tiers). SGLang keeps the router, the shared expert and everything else on the GPU.

Two submission paths, same worker, same numerics:
  * decode-size batches (<= SGLANG_EXL3_CPU_MOE_DEV_ROWS rows, default 32): device-driven handoff
    (csrc/offload/offload_kernels.cu: cpu_moe_issue / cpu_moe_collect). The GPU owns the job sequence counter, writes
    the job descriptor and stages the inputs itself, so the whole step is capturable in SGLang's decode CUDA graphs.
  * larger batches (prefill chunks, eager only): `MoeCpuHost.submit_prefill` unchanged -- experts with >= stream_t
    assigned rows are DMA'd to the GPU through the worker's pinned staging ring and computed there, the cold tail
    runs on the CPU concurrently. Before it, the stream is synchronized and the device-owned sequence counter handed
    to the host object; afterwards the host's counter is written back for the device path.

Env: SGLANG_EXL3_MOE_OFFLOAD=cpu selects this method (plugin/config); EXL3_MOE_CPU_THREADS (worker threads) and the
other EXL3_MOE_* knobs of exllamav3's MoeCpuTuning apply unchanged.
"""
from __future__ import annotations

import logging
import os
import re
import time
import types

import numpy as np
import torch
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils.common import set_weight_attrs

logger = logging.getLogger(__name__)
_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")
_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_KEY_RE = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
DEV_ROWS = int(os.environ.get("SGLANG_EXL3_CPU_MOE_DEV_ROWS", "32"))
_TIMEOUT_NS = int(float(os.environ.get("SGLANG_EXL3_CPU_MOE_TIMEOUT_S", "120")) * 1e9)

_DEBUG = os.environ.get("SGLANG_EXL3_CPU_MOE_DEBUG", "0") == "1"
# diagnostic only: routed experts contribute zero (no CPU work) -> measures the non-expert decode/prefill cost
_NULL = os.environ.get("SGLANG_EXL3_MOE_NULL", "0") == "1"
_METHODS: list = []          # every Exl3CpuMoEMethod built for this model (registration count)
_STATE = {"host": None, "registered": 0, "dev": None}


def _host(model_path: str):
    if _STATE["host"] is None:
        from exllamav3.model.moe_cpu_host import MoeCpuHost
        threads = int(os.environ.get("EXL3_MOE_CPU_THREADS", "0")) or None
        ip = types.SimpleNamespace(moe_cpu_component="text", moe_cpu_threads=threads)
        cfg = types.SimpleNamespace(directory=model_path, infer_params=ip)
        _STATE["host"] = MoeCpuHost(cfg)
    return _STATE["host"]


def _dbg_wait(tag, limit=15.0):
    """Debug only (eager): bounded wait for the stream, then print the shared control words."""
    from exllamav3.model import moe_cpu_host as mch
    host = _STATE["host"]
    ev = torch.cuda.Event()
    ev.record()
    t0 = time.time()
    while not ev.query() and time.time() - t0 < limit:
        time.sleep(0.002)
    u = np.frombuffer(host.shm.buf, dtype=np.uint32)
    f = mch.MOE_SLOT_FLAGS_OFFSET // 4
    print(f"[cpu_moe dbg] {tag}: {'ok' if ev.query() else 'STUCK'} {time.time() - t0:.3f}s devseq {u[1]} abort {u[32]} "
          f"tail {u[64]} head {u[80]} ready0 {u[f]} done0 {u[f + 128]} cons0 {u[f + 256]} host.seq {host.seq}", flush=True)


class _DevPath:
    """Addresses of compute slot 0 + control block in the worker's registered shared region (device view)."""

    def __init__(self, host, device):
        from exllamav3.model import moe_cpu_host as mch
        from . import _ext
        self.ext = _ext.load()
        s0 = host.slots[0]
        self.base = host.gpu_base_ptr
        self.jobs = self.base + mch.MOE_CTRL_JOBS_OFFSET
        self.job_words = mch.MOE_JOB_BYTES // 4
        self.ring = mch.MOE_JOB_RING
        self.x, self.sel, self.w, self.out = s0["x_dev"], s0["sel_dev"], s0["w_dev"], s0["out_dev"]
        self.data_ready, self.done, self.consumed = s0["data_ready"], s0["done"], s0["consumed"]
        self.hi, self.ho = host.layout["max_hi"], host.layout["max_ho"]
        self.cap_rows = host.layout["cap_rows"]
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        # host view of the device-owned sequence counter (ctrl + 4 bytes)
        self.devseq = np.frombuffer(host.shm.buf, dtype=np.uint32, count=1, offset=4)
        self.devseq[0] = host.seq


class Exl3CpuMoEMethod(FusedMoEMethodBase):
    def __init__(self, config, prefix: str, experts: dict[str, tuple]):
        self.config = config
        self.prefix = prefix
        self.runner = None
        self.moe_runner_config = None
        self.keys: dict[tuple[str, int], str] = {}
        self.infos: dict[tuple[str, int], tuple] = {}
        for key, info in experts.items():
            m = _KEY_RE.search(key)
            if m is None:
                raise ValueError(f"{prefix}: unexpected EXL3 matrix under the experts: {key}")
            rk = (m.group(2)[:-5], int(m.group(1)))
            self.keys[rk], self.infos[rk] = key, info
        _METHODS.append(self)

    # ---- weights: only the per-expert scale vectors are kept (GPU, for the streamed-prefill dequant); the trellis
    # words are read by the worker process from the checkpoint itself
    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        if getattr(layer, "moe_tp_size", 1) != 1 or getattr(layer, "moe_ep_size", 1) != 1:
            raise NotImplementedError(f"{self.prefix}: TP/EP not supported by the CPU expert offload")
        if getattr(layer, "num_fused_shared_experts", 0):
            raise NotImplementedError(f"{self.prefix}: run with --disable-shared-experts-fusion")
        expected = {(role, e) for role in ("gate", "up", "down") for e in range(num_experts)}
        missing = sorted(expected - set(self.infos))
        if missing:
            raise ValueError(f"{self.prefix}: no EXL3 tensors for {len(missing)} expert matrices, e.g. {missing[0]}")
        if any(self.infos[k][4] for k in expected):
            raise NotImplementedError(f"{self.prefix}: EXL3 experts with bias are not implemented")
        if any(self.infos[k][3] != "mul1" for k in expected):
            raise NotImplementedError(f"{self.prefix}: the CPU worker runs mul1 experts only")
        layer.exl3_seen = set()
        layer.exl3_aux = {}
        layer.exl3_num_experts = num_experts
        layer.exl3_hidden = hidden_size
        layer.exl3_inter = intermediate_size_per_partition
        for w in ("w13", "w2"):
            for suffix in _SUFFIXES:
                p = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": self._make_loader(layer, suffix), "exl3_placeholder": True})
                layer.register_parameter(f"{w}_{suffix}", p)

    def _make_loader(self, layer, suffix):
        def load(param, loaded_weight, weight_name=None, shard_id=None, expert_id=None, *a, **k):
            role = _ROLE.get(shard_id)
            if role is None or expert_id is None:
                raise ValueError(f"{self.prefix}.{suffix}: unexpected expert shard {shard_id!r} / expert {expert_id!r}")
            e = int(expert_id)
            if suffix in ("su", "sv"):
                raise NotImplementedError(f"{self.prefix}: legacy packed-sign experts are not supported by the CPU worker")
            if suffix in ("suh", "svh"):
                dev = torch.device("cuda", torch.cuda.current_device())
                layer.exl3_aux[(role, suffix, e)] = loaded_weight.to(dev, torch.float16, copy=True)
            elif suffix == "trellis":
                layer.exl3_seen.add((role, e))
        return load

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = cfg = moe_runner_config
        if cfg.activation != "silu" or not getattr(cfg, "is_gated", True):
            raise NotImplementedError(f"{self.prefix}: only SiLU-gated experts are implemented (got {cfg.activation})")
        if cfg.apply_router_weight_on_input:
            raise NotImplementedError(f"{self.prefix}: apply_router_weight_on_input is not supported")
        if cfg.routed_scaling_factor not in (None, 1.0):
            raise NotImplementedError(f"{self.prefix}: routed_scaling_factor {cfg.routed_scaling_factor} is not supported")

    def process_weights_after_loading(self, layer) -> None:
        n = layer.exl3_num_experts
        lost = [(r, e) for r in ("gate", "up", "down") for e in range(n) if (r, e) not in layer.exl3_seen]
        if lost:
            raise ValueError(f"{self.prefix}: {len(lost)} expert matrices never delivered, e.g. {lost[0]}")
        aux = {}
        for r, p in (("gate", "g"), ("up", "u"), ("down", "d")):
            for s in ("suh", "svh"):
                lst = [layer.exl3_aux.get((r, s, e)) for e in range(n)]
                if any(t is None for t in lst):
                    raise ValueError(f"{self.prefix}: {r}_proj {s} missing for some experts")
                aux[f"{s}_{p}"] = lst
            aux[f"bias_{p}"] = None
        del layer.exl3_aux, layer.exl3_seen
        for w in ("w13", "w2"):
            for suffix in _SUFFIXES:
                delattr(layer, f"{w}_{suffix}")

        def dims(role):
            k, nn_, twice = self.infos[(role, 0)][:3]
            return (k, nn_, twice // 2)
        pd = dict(g=dims("gate"), u=dims("up"), d=dims("down"))
        for role in ("gate", "up", "down"):
            if len({self.infos[(role, e)][:3] for e in range(n)}) != 1:
                raise NotImplementedError(f"{self.prefix}: {role}_proj experts differ in shape / K")
        top_k = self.moe_runner_config.top_k if self.moe_runner_config is not None else getattr(layer, "top_k")
        host = _host(self.config.model_path)
        layer.exl3_cpu_idx = host.register_layer(
            self.prefix,
            [self.keys[("gate", e)] for e in range(n)],
            [self.keys[("up", e)] for e in range(n)],
            [self.keys[("down", e)] for e in range(n)],
            0, 0.0, pd["u"][0], pd["d"][1], int(top_k), proj_dims=pd, aux=aux)
        _STATE["registered"] += 1
        logger.info("%s: %d routed experts -> CPU worker (layer %d, K %g/%g/%g)", self.prefix, n, layer.exl3_cpu_idx,
                    pd["g"][2], pd["u"][2], pd["d"][2])
        if _STATE["registered"] == len(_METHODS):
            t0 = time.time()
            host.ensure_started()
            _STATE["dev"] = _DevPath(host, torch.device("cuda", torch.cuda.current_device()))
            logger.info("CPU MoE worker ready: %d layers, %d threads (%.1f s after the last registration)",
                        len(host.specs), host.threads, time.time() - t0)
            self._warm(layer)

    @torch.no_grad()
    def _warm(self, layer):
        dev = torch.device("cuda", torch.cuda.current_device())
        top_k = self.moe_runner_config.top_k if self.moe_runner_config is not None else 10
        for t in (1, DEV_ROWS, 256):
            x = torch.randn((t, layer.exl3_hidden), dtype=torch.bfloat16, device=dev) * 0.1
            ids = torch.randint(0, layer.exl3_num_experts, (t, top_k), dtype=torch.int32, device=dev)
            w = torch.full((t, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
            y = self._experts(layer, x, ids, w)
            torch.cuda.synchronize(dev)
            if not torch.isfinite(y).all():
                raise RuntimeError(f"{self.prefix}: CPU MoE warm-up produced non-finite output ({t} rows)")
        host = _STATE["host"]
        if host.v_abort[0]:
            raise RuntimeError("CPU MoE worker abort flag set during warm-up")

    # ---- forward
    def _experts(self, layer, x, ids, w):
        rows = x.shape[0]
        if rows == 0 or _NULL:
            return torch.zeros_like(x)
        d = _STATE["dev"]
        if rows <= min(DEV_ROWS, d.cap_rows):
            return self._device_path(layer, x, ids, w)
        from sglang.srt.model_executor.runner import get_is_capture_mode
        if get_is_capture_mode():
            raise RuntimeError(f"{self.prefix}: {rows}-row batch in a CUDA graph capture exceeds the device-driven CPU "
                               f"MoE path ({DEV_ROWS} rows); lower --cuda-graph-max-bs")
        return self._host_path(layer, x, ids, w)

    def _device_path(self, layer, x, ids, w):
        d = _STATE["dev"]
        x = x.contiguous()
        if _DEBUG and not torch.cuda.is_current_stream_capturing():
            _dbg_wait(f"L{layer.exl3_cpu_idx} before issue rows {x.shape[0]} x finite {bool(torch.isfinite(x).all())} "
                      f"absmax {float(x.float().abs().max()):.3g} ids [{int(ids.min())},{int(ids.max())}] "
                      f"w finite {bool(torch.isfinite(w).all())}")
        d.ext.cpu_moe_issue(x, ids.contiguous(), w.to(torch.float32).contiguous(), d.hi, d.x, d.sel, d.w, d.base,
                            d.jobs, d.job_words, d.ring, d.data_ready, layer.exl3_cpu_idx, d.counter)
        if _DEBUG and not torch.cuda.is_current_stream_capturing():
            _dbg_wait("after issue")
        out = torch.empty_like(x)
        d.ext.cpu_moe_collect(out, d.out, d.ho, d.base, d.done, d.consumed, _TIMEOUT_NS)
        if _DEBUG and not torch.cuda.is_current_stream_capturing():
            _dbg_wait(f"after collect: out finite {bool(torch.isfinite(out).all())} absmax {float(out.float().abs().max()):.3g}")
        return out

    def _host_path(self, layer, x, ids, w):
        host, d = _STATE["host"], _STATE["dev"]
        torch.cuda.current_stream().synchronize()          # device-driven jobs done: counters are current
        host.seq = int(d.devseq[0])
        host.slot_last_seq = [0] * len(host.slot_last_seq)
        host.begin_pass()
        out = host.submit_prefill(layer.exl3_cpu_idx, x.to(torch.float16).contiguous(), ids.to(torch.long),
                                  w.to(torch.float16).contiguous())
        d.devseq[0] = host.seq                              # read by the next device-driven issue (stream order)
        return out.to(x.dtype)

    def apply(self, layer, dispatch_output):
        x = dispatch_output.hidden_states
        tk = dispatch_output.topk_output
        flat = x.reshape(-1, x.shape[-1])
        y = self._experts(layer, flat, tk.topk_ids.reshape(flat.shape[0], -1), tk.topk_weights.reshape(flat.shape[0], -1))
        return StandardCombineInput(hidden_states=y.view_as(x))

    def get_triton_quant_info(self, layer):
        raise NotImplementedError("EXL3 CPU-offloaded experts do not run on the Triton MoE runner")
