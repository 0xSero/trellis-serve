"""Exl3MoEMethod: EXL3 routed experts for SGLang's FusedMoE (Qwen3.6-35B-A3B and its MTP draft layer).

Every expert keeps its own checkpoint tensors (trellis, suh, svh), nothing is concatenated: SGLang's model
`load_weights` maps `experts.E.{gate,up,down}_proj.<suffix>` to parameters named `experts.w13_<suffix>` (shard ids
w1 = gate, w3 = up) and `experts.w2_<suffix>` (w2 = down) and calls `param.weight_loader(param, tensor, name,
shard_id=, expert_id=)`. Placeholders with those names are registered here (every suffix an EXL3 checkpoint can
carry, so no expert tensor ever raises KeyError in the loader) and the loader files the tensors under
(projection, suffix, expert). `process_weights_after_loading` checks completeness, then builds either

  * the reference path: per-expert tensors + int64 pointer tables for ExLlamaV3's `exl3_mgemm` (three launches:
    gate, up, weighted down-reduction; kernels/reference.moe_mgemm), one K and one codebook per projection; or
  * the grouped path (`SGLANG_EXL3_MOE_KERNEL=auto|marlin`, module `trellis_exl3_moe_kernels`, K = 3 / 4, mcg / mul1):
    stacked [E, ...] packs (a pure re-layout of the stored words) run by kernels/marlin_moe (5 launches + SGLang's
    `moe_align_block_size`). Then the per-expert copies are released: the experts are resident once.

Both are CUDA-graph safe: shapes depend only on (tokens, top_k); the reference path's mgemm autotune (per
(k, n, K, codebook, m-bucket) key, m = 1 per slot here, so one key per projection shape for every batch size) is
run once at load, before any capture. TP = 1, EP = 1, SiLU-gated experts, no fused shared expert (the shared expert
stays an ordinary EXL3 linear: `--disable-shared-experts-fusion`, `Exl3Config.can_fuse_shared_expert() -> False`).
"""
from __future__ import annotations

import logging
import os
import re

import torch
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils.common import set_weight_attrs

from ..kernels import marlin_moe, reference
from .linear import _unpack_signs

logger = logging.getLogger(__name__)
_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")
_CODEBOOK_ID = {"3inst": 0, "mcg": 1, "mul1": 2}
_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_PROJ_NAME = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
_KEY_RE = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
# auto | exllamav3 | marlin; defaults to SGLANG_EXL3_KERNEL so one switch covers dense and experts
_MOE_KERNEL = os.environ.get("SGLANG_EXL3_MOE_KERNEL", os.environ.get("SGLANG_EXL3_KERNEL", "auto"))
_KEEP_REFERENCE = os.environ.get("SGLANG_EXL3_MOE_KEEP_REFERENCE", "0") == "1"   # keep mgemm tables next to the pack
_WARM_TOKENS = int(os.environ.get("SGLANG_EXL3_MOE_WARM_TOKENS", "8"))


def _align(topk_ids: torch.Tensor, block: int, num_experts: int):
    """SGLang's moe_align_block_size (sgl_kernel / triton): vLLM's conventions (sorted slot ids padded with
    topk_ids.numel(), one expert id per block, num_tokens_post_padded on the device), graph-safe."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    return moe_align_block_size(topk_ids, block, num_experts)


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, config, prefix: str, experts: dict[str, tuple]):
        self.config = config
        self.prefix = prefix                    # "...mlp.experts"
        self.runner = None                      # FusedMoE reads quant_method.runner; the MoE runner stack is bypassed
        self.moe_runner_config = None
        # header scan: (role, expert id) -> (k, n, 2K, codebook, has_bias, legacy_signs)
        self.infos: dict[tuple[str, int], tuple] = {}
        for key, info in experts.items():
            m = _KEY_RE.search(key)
            if m is None:
                raise ValueError(f"{prefix}: unexpected EXL3 matrix under the experts: {key}")
            self.infos[(m.group(2)[:-5], int(m.group(1)))] = info

    # ---- weights
    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        if getattr(layer, "moe_tp_size", 1) != 1 or getattr(layer, "moe_ep_size", 1) != 1:
            raise NotImplementedError(f"{self.prefix}: tensor/expert-parallel EXL3 MoE is not implemented (single GPU only)")
        if getattr(layer, "num_fused_shared_experts", 0):
            raise NotImplementedError(f"{self.prefix}: fused shared experts are not supported by the EXL3 MoE method; "
                                      f"run with --disable-shared-experts-fusion")
        if extra_weight_attrs.get("with_bias"):
            raise NotImplementedError(f"{self.prefix}: EXL3 experts with bias are not implemented")
        expected = {(role, e) for role in ("gate", "up", "down") for e in range(num_experts)}
        missing = sorted(expected - set(self.infos))
        if missing:
            raise ValueError(f"{self.prefix}: the checkpoint has no EXL3 tensors for {len(missing)} of {3 * num_experts} "
                             f"expert matrices, e.g. {missing[0]}")
        if any(self.infos[k][4] for k in expected):
            raise NotImplementedError(f"{self.prefix}: EXL3 experts with bias are not implemented")
        layer.exl3_store = {}                   # (role, suffix) -> {expert id: tensor (CPU copy)}
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
            store = layer.exl3_store.setdefault((role, suffix), {})
            e = int(expert_id)
            if e in store:
                raise ValueError(f"{self.prefix}: {role} {suffix} of expert {e} loaded twice")
            # CPU copy until every expert is in: the GPU then receives one pack (or the pointer-table tensors) per
            # layer instead of 768 small tensors it has to keep and repack next to.
            store[e] = loaded_weight.detach().to("cpu", copy=True)
        return load

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        cfg = moe_runner_config
        if cfg.activation != "silu" or not getattr(cfg, "is_gated", True):
            raise NotImplementedError(f"{self.prefix}: only SiLU-gated experts are implemented (got {cfg.activation})")
        if cfg.apply_router_weight_on_input:
            raise NotImplementedError(f"{self.prefix}: apply_router_weight_on_input is not supported")
        if cfg.routed_scaling_factor not in (None, 1.0):
            raise NotImplementedError(f"{self.prefix}: routed_scaling_factor {cfg.routed_scaling_factor} is not supported")
        if getattr(cfg, "num_fused_shared_experts", 0):
            raise NotImplementedError(f"{self.prefix}: fused shared experts are not supported (--disable-shared-experts-fusion)")

    def process_weights_after_loading(self, layer) -> None:
        store, n = layer.exl3_store, layer.exl3_num_experts
        dev = torch.device("cuda", torch.cuda.current_device())
        books, bits = set(), {}
        per: dict[str, tuple[list, list, list]] = {}
        for role in ("gate", "up", "down"):
            tensors = {s: store.get((role, s), {}) for s in _SUFFIXES}
            missing = [e for e in range(n) if e not in tensors["trellis"]]
            if missing:
                raise ValueError(f"{self.prefix}: {role}_proj trellis missing for {len(missing)} experts, e.g. {missing[0]}")
            trellis, suh, svh = [], [], []
            for e in range(n):
                t = tensors["trellis"][e]
                su, sv = tensors["suh"].get(e), tensors["svh"].get(e)
                if su is None and e in tensors["su"]:
                    su, sv = _unpack_signs(tensors["su"][e]), _unpack_signs(tensors["sv"][e])
                if su is None or sv is None:
                    raise ValueError(f"{self.prefix}: {role}_proj of expert {e} has a trellis but no scale vectors")
                codebook = "mcg" if e in tensors["mcg"] else "mul1" if e in tensors["mul1"] else "3inst"
                k, nn_, twice, declared = self.infos[(role, e)][:4]
                if (t.shape[0] * 16, t.shape[1] * 16, t.shape[2] // 8, codebook) != (k, nn_, twice, declared):
                    raise ValueError(f"{self.prefix}: {role}_proj expert {e} {tuple(t.shape)}/{codebook} does not match "
                                     f"the header scan {(k, nn_, twice, declared)}")
                books.add(codebook)
                trellis.append(t.contiguous()); suh.append(su.contiguous()); svh.append(sv.contiguous())
            if len({tuple(t.shape) for t in trellis}) != 1:
                raise NotImplementedError(f"{self.prefix}: experts of {role}_proj differ in shape / bitrate (mixed K)")
            bits[role] = trellis[0].shape[2] / 16
            per[role] = (trellis, suh, svh)
        if len(books) != 1:
            raise NotImplementedError(f"{self.prefix}: mixed codebooks {books}")
        if bits["gate"] != bits["up"]:
            raise NotImplementedError(f"{self.prefix}: gate and up bitrates differ")
        kt, nt = per["gate"][0][0].shape[:2]
        if kt * 16 != layer.exl3_hidden or nt * 16 < layer.exl3_inter or per["down"][0][0].shape[:2] != (nt, kt):
            raise ValueError(f"{self.prefix}: expert shapes {tuple(per['gate'][0][0].shape)} / "
                             f"{tuple(per['down'][0][0].shape)} do not fit hidden {layer.exl3_hidden}, "
                             f"intermediate {layer.exl3_inter}")
        if nt * 16 != layer.exl3_inter:
            raise NotImplementedError(f"{self.prefix}: padded intermediate ({nt * 16} stored vs {layer.exl3_inter})")
        layer.exl3_codebook = _CODEBOOK_ID[books.pop()]
        layer.exl3_bits = bits
        store.clear()
        del layer.exl3_store
        for w in ("w13", "w2"):
            for suffix in _SUFFIXES:
                delattr(layer, f"{w}_{suffix}")

        # ---- grouped kernel (stacked packs), else the reference pointer tables
        layer.exl3_pack = None
        layer.exl3_tables = None
        why = ""
        if _MOE_KERNEL in ("auto", "marlin"):
            status = marlin_moe.probe()
            ok, why = (marlin_moe.supports(per["gate"][0], per["up"][0], per["down"][0], layer.exl3_codebook)
                       if status.available else (False, status.detail))
            if ok:
                pack = marlin_moe.prepare(per["gate"], per["up"], per["down"], layer.exl3_codebook)   # built on the CPU, moved once
                for name, t in zip(("w13", "w2", "suh13", "svh13", "suh2", "svh2"), pack.tensors()):
                    layer.register_buffer(f"exl3_pack_{name}", t, persistent=False)
                layer.exl3_pack = pack
            elif _MOE_KERNEL == "marlin":
                raise RuntimeError(f"{self.prefix}: SGLANG_EXL3_MOE_KERNEL=marlin but the grouped kernel cannot serve "
                                   f"this layer ({why})")
        if layer.exl3_pack is None or _KEEP_REFERENCE:
            tables = {}
            for role in ("gate", "up", "down"):
                keep = [[t.to(dev) for t in group] for group in per[role]]
                setattr(layer, f"exl3_{role}_tensors", keep)             # keeps the storage alive and in place
                tables[role] = tuple(reference.pointer_table(g) for g in keep)
                for j, s in enumerate(("trellis", "suh", "svh")):
                    layer.register_buffer(f"exl3_{role}_{s}_ptrs", tables[role][j], persistent=False)
            layer.exl3_tables = tables
        del per
        torch.cuda.empty_cache()
        logger.info("%s: %d experts, K gate/up %g down %g, codebook %d -> %s", self.prefix, n, bits["gate"], bits["down"],
                    layer.exl3_codebook, "grouped kernel (trellis_exl3_moe_kernels)" if layer.exl3_pack is not None
                    else f"ExLlamaV3 exl3_mgemm ({why or _MOE_KERNEL})")
        self._warm(layer, dev)

    @torch.no_grad()
    def _warm(self, layer, dev) -> None:
        """Run the serving path once per configured path: ExLlamaV3's mgemm autotunes on first use per (k, n, K, cb,
        m-bucket) and allocates its device context (both would break a CUDA-graph capture); the grouped kernel
        initialises its per-device state in prepare, its align/triton kernels compile here."""
        n, top_k = layer.exl3_num_experts, (self.moe_runner_config.top_k if self.moe_runner_config else 8)
        tokens = max(1, _WARM_TOKENS)
        x = torch.zeros((tokens, layer.exl3_hidden), dtype=torch.bfloat16, device=dev)
        ids = (torch.arange(tokens * top_k, device=dev).view(tokens, top_k) * 37) % n
        w = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=dev)
        for t in (tokens, 1):
            self._experts(layer, x[:t], ids[:t].to(torch.int32).contiguous(), w[:t])
        torch.cuda.synchronize(dev)

    # ---- forward
    def _experts(self, layer, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        pack = layer.exl3_pack
        if pack is not None:
            tokens = x.shape[0]
            if tokens == 0:
                return torch.empty_like(x)
            ids = topk_ids.contiguous()
            block = marlin_moe.moe_block_size(tokens, ids.shape[1], pack.num_experts)   # shapes only: graph-safe
            routing = _align(ids, block, pack.num_experts)
            return marlin_moe.run(x.contiguous(), topk_weights.to(torch.float32).contiguous(), ids, *routing, block, pack)
        tab = layer.exl3_tables
        return reference.moe_mgemm(x, topk_ids, topk_weights, tab["gate"], tab["up"], tab["down"],
                                   layer.exl3_bits["gate"], layer.exl3_bits["down"], layer.exl3_inter,
                                   layer.exl3_num_experts, layer.exl3_codebook)

    def apply(self, layer, dispatch_output):
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        flat = x.reshape(-1, x.shape[-1])
        y = self._experts(layer, flat, topk_ids.reshape(flat.shape[0], -1), topk_weights.reshape(flat.shape[0], -1))
        return StandardCombineInput(hidden_states=y.view_as(x))

    def get_triton_quant_info(self, layer):
        raise NotImplementedError("EXL3 experts do not run on the Triton MoE runner")
