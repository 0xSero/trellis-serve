"""Vision-tower attention for Ampere (opt-in, SGLANG_EXL3_VIT_SDPA=1).

On sm_86 SGLang's default multimodal attention is `triton_attn` (its generic `context_attention_fwd`); the ViT of
Qwen3.5/3.8 (head_dim 72, full attention over every patch of an image) runs it far below the card's attention rate.
SGLang's own `sdpa` backend materialises an s x s boolean mask for cu_seqlens (4.3 GB for one 4096x4096 image) and so
cannot reach PyTorch's flash kernel. This backend runs one mask-free `scaled_dot_product_attention` per image
segment, restricted to the flash / memory-efficient kernels (never the O(s^2)-memory math fallback). Same math
(non-causal softmax attention within each cu_seqlens segment)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


class SegmentSdpaAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, q, k, v, cu_seqlens, bsz, seq_len, softmax_scale=None, forward_metadata=None, **kwargs):
        from sglang.srt.layers.attention import vision as V
        if forward_metadata is not None:
            cu = forward_metadata.cu_seqlens
        elif isinstance(cu_seqlens, list):          # SGLANG_VIT_ENABLE_CUDA_GRAPH layout
            cu = cu_seqlens[0]
        else:
            cu = V.resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
        bounds = [int(x) for x in cu.tolist()]
        out = kwargs.get("output_ws")
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        with sdpa_kernel(_BACKENDS):
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b <= a:
                    continue
                # [L, h, d] -> [1, h, L, d]
                qs, ks, vs = (t[a:b].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
                o = F.scaled_dot_product_attention(qs, ks, vs, scale=softmax_scale)
                out[a:b] = o[0].transpose(0, 1)
        return out


def install() -> None:
    from sglang.srt.layers.attention import vision as V
    V.QKV_BACKEND_IMPL["triton_attn"] = SegmentSdpaAttention


def install_mlp_chunk(rows: int) -> None:
    """Row-chunked ViT MLP (opt-in, SGLANG_EXL3_VIT_MLP_CHUNK=<rows>). The MLP is row-wise, so chunking over patches is
    the same math; it bounds the fc1 + activation transients (2 x 538 MiB for one 4096x4096 image: 65,536 patches x
    4,304 x bf16) to rows x 4,304 x 2 x 2 bytes, which is what let a 4096^2 image through next to a C4 + MTP KV pool."""
    from sglang.srt.models import qwen3_vl as m
    orig = m.Qwen3_VisionMLP.forward

    def forward(self, x):
        n = x.shape[0]
        if n <= rows:
            return orig(self, x)
        out = None
        for a in range(0, n, rows):
            y = orig(self, x[a:a + rows])
            if out is None:
                out = y.new_empty((n, *y.shape[1:]))
            out[a:a + y.shape[0]] = y
        return out

    m.Qwen3_VisionMLP.forward = forward


def install_fast_processor_cpu() -> None:
    """Run the Transformers fast (torchvision) image processor on CPU (opt-in, SGLANG_EXL3_MM_FAST_CPU=1). SGLang places
    it on the serving GPU, where a 4096x4096 image's intermediate tensors OOM next to a full KV pool on 24 GB; the PIL
    backend fits but preprocesses in single-threaded numpy."""
    from sglang.srt.multimodal.processors import base_processor as b
    b.BaseMultimodalProcessor._fast_image_processor_device = lambda self, processor: "cpu"
