// Offload kernels for trellis-serve (Qwen3.8-Flash-Next on one 24 GB card):
//
//  * cpu_moe_issue / cpu_moe_collect: drive exllamav3's persistent CPU MoE worker (moe_cpu_host.py,
//    cpu/moe_handoff.h) entirely from the GPU stream, so the routed-expert handoff is CUDA-graph capturable.
//    The stock host path writes the job descriptor and the sequence number from the host at enqueue time; a graph
//    replay would repeat stale values. Here the GPU owns the sequence counter (u32 at ctrl+4 in the shared
//    region), writes the descriptor into the job ring, bumps jobs_tail, stages x / sel / w into compute slot 0
//    with zero-copy stores and publishes data_ready; collect spins on done[0] and reads the fp32 partial back.
//    Slot 0 is safe to reuse for every device-driven job: each job is collected (done observed) on the same
//    stream before the next one is issued, and the worker never touches a slot after publishing done.
//
//  * ngram_gather_dequant: the Flash-Next n-gram table (exl3_ngram_trellis: per row 1 fp16 scale word + 160 x K bit
//    tail-biting mul1 trellis) lives in pinned, mapped host memory; one block per looked-up row reads the packed
//    row zero-copy and decodes it (same math as exllamav3 ngram_dequant_kernel, + per-head fp16 bias).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t ld_sys(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_sys(uint32_t* p, uint32_t v)
{
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<at::Half>(at::Half v) { return __half2float((__half) v); }
template <> __device__ __forceinline__ float to_f<at::BFloat16>(at::BFloat16 v) { return __bfloat162float((__nv_bfloat16) v); }
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ at::Half from_f<at::Half>(float v) { return (at::Half) __float2half_rn(v); }
template <> __device__ __forceinline__ at::BFloat16 from_f<at::BFloat16>(float v) { return (at::BFloat16) __float2bfloat16_rn(v); }

// ctrl block offsets (u32 index) -- mirror moe_handoff.h / moe_cpu_host.py
#define CTRL_DEVSEQ 1      // byte 4: device-owned job sequence counter (ours; the worker never reads it)
#define CTRL_ABORT 32      // byte 128
#define CTRL_TAIL 64       // byte 256

template <typename TX, typename TI>
__global__ void cpu_moe_issue_kernel
(
    const TX* __restrict__ x, int64_t x_stride,
    const TI* __restrict__ ids,
    const float* __restrict__ w,
    int rows, int h, int hi, int topk,
    __half* sx, int32_t* ssel, __half* sw,
    uint32_t* ctrl, uint32_t* jobs, int job_words, int ring,
    uint32_t* data_ready, uint32_t layer, int* counter
)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    for (int i = t; i < hi; i += blockDim.x)
        sx[(size_t) r * hi + i] = __float2half_rn(i < h ? to_f<TX>(x[(size_t) r * x_stride + i]) : 0.0f);
    if (t < topk)
    {
        ssel[r * topk + t] = (int32_t) ids[(size_t) r * topk + t];
        sw[r * topk + t] = __float2half_rn(w[(size_t) r * topk + t]);
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0)
    {
        int prev = atomicAdd(counter, 1);
        if (prev == (int) gridDim.x - 1)
        {
            // last block: every row is visible system-wide; publish descriptor, tail, data flag in that order
            atomicExch(counter, 0);
            uint32_t seq = ld_sys(ctrl + CTRL_DEVSEQ) + 1;
            st_sys(ctrl + CTRL_DEVSEQ, seq);
            uint32_t tail = ld_sys(ctrl + CTRL_TAIL);
            volatile uint32_t* job = jobs + (size_t) (tail % (uint32_t) ring) * job_words;
            job[0] = seq; job[1] = layer; job[2] = (uint32_t) rows; job[3] = (uint32_t) topk;
            job[4] = 0; job[5] = 0; job[6] = 0;     // slot 0, MOE_JOB_KIND_COMPUTE
            __threadfence_system();
            st_sys(ctrl + CTRL_TAIL, tail + 1);
            __threadfence_system();
            st_sys(data_ready, seq);
        }
    }
}

template <typename TO>
__global__ void cpu_moe_collect_kernel
(
    TO* __restrict__ out, int64_t out_stride,
    const float* sout, int rows, int h, int ho,
    uint32_t* ctrl, const uint32_t* done, uint32_t* consumed, long long timeout_ns
)
{
    __shared__ int s_ok;
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    if (t == 0)
    {
        uint32_t seq = ld_sys(ctrl + CTRL_DEVSEQ);
        long long waited = 0;
        unsigned sleep = 32;
        int ok = 1;
        while ((int32_t) (ld_sys(done) - seq) < 0)
        {
            __nanosleep(sleep);
            waited += sleep;
            if (sleep < 1024) sleep <<= 1;
            if (waited > timeout_ns) { st_sys(ctrl + CTRL_ABORT, 1); ok = 0; break; }
        }
        s_ok = ok;
        if (r == 0) st_sys(consumed, seq);
    }
    __syncthreads();
    const volatile float* src = sout + (size_t) r * ho;
    for (int i = t; i < h; i += blockDim.x)
        out[(size_t) r * out_stride + i] = from_f<TO>(s_ok ? src[i] : 0.0f);
}

#define ROW_DIM 160
#define MUL1 0x83DCD12Du

template <typename TO>
__global__ void ngram_gather_dequant_kernel
(
    const int64_t* __restrict__ ids,      // (N) global row ids
    const int16_t* table,                 // (rows, words) host-mapped
    int64_t num_rows,
    const __half* __restrict__ bias,      // (num_heads, ROW_DIM)
    TO* __restrict__ out,                 // (N, ROW_DIM)
    int K, int words, int num_heads
)
{
    const int r = blockIdx.x;
    const int i = threadIdx.x;
    extern __shared__ uint16_t sw[];
    int64_t row = ids[r];
    bool valid = row >= 0 && row < num_rows;
    if (i < words) sw[i] = valid ? (uint16_t) ((const volatile int16_t*) table)[row * words + i] : 0;
    __syncthreads();
    if (i >= ROW_DIM) return;
    float scale = __half2float(__ushort_as_half(sw[0]));
    uint32_t state = 0;
    #pragma unroll 4
    for (int m = 0; m < 16; ++m)
    {
        int pos = i - m / K;
        if (pos < 0) pos += ROW_DIM;
        int sb = pos * K + m % K;
        uint32_t bit = (sw[1 + (sb >> 4)] >> (sb & 15)) & 1;
        state |= bit << m;
    }
    uint32_t prod = state * MUL1;
    float hsum = 1024.0f + (float) ((prod & 0xff) + ((prod >> 8) & 0xff) + ((prod >> 16) & 0xff) + ((prod >> 24) & 0xff));
    float k_inv = __half2float(__ushort_as_half((unsigned short) 0x1eee));
    float k_bias = __half2float(__ushort_as_half((unsigned short) 0xc931));
    float cb = __half2float(__float2half_rn(hsum * k_inv + k_bias));
    int head = r % num_heads;
    float b = __half2float(bias[(size_t) head * ROW_DIM + i]);
    // round through fp16 like the reference (fp16 rows), then store in the output dtype
    float v = __half2float(__float2half_rn(cb * scale + b));
    out[(size_t) r * ROW_DIM + i] = from_f<TO>(v);
}

} // namespace

void cpu_moe_issue
(
    const at::Tensor& x, const at::Tensor& ids, const at::Tensor& w,
    int64_t hi, int64_t sx, int64_t ssel, int64_t sw,
    int64_t ctrl, int64_t jobs, int64_t job_words, int64_t ring,
    int64_t data_ready, int64_t layer, at::Tensor& counter
)
{
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "cpu_moe_issue: x must be 2-D, unit inner stride");
    TORCH_CHECK(ids.is_contiguous() && w.is_contiguous() && w.scalar_type() == at::kFloat, "cpu_moe_issue: ids/w layout");
    TORCH_CHECK(counter.scalar_type() == at::kInt && counter.is_cuda(), "cpu_moe_issue: counter");
    int rows = (int) x.size(0);
    if (rows == 0) return;
    int topk = (int) ids.size(1);
    TORCH_CHECK(topk <= 128, "cpu_moe_issue: topk");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    #define ISSUE(TX, TI) cpu_moe_issue_kernel<TX, TI><<<rows, threads, 0, stream>>>( \
        (const TX*) x.data_ptr(), x.stride(0), (const TI*) ids.data_ptr(), w.data_ptr<float>(), \
        rows, (int) x.size(1), (int) hi, topk, (__half*) sx, (int32_t*) ssel, (__half*) sw, \
        (uint32_t*) ctrl, (uint32_t*) jobs, (int) job_words, (int) ring, (uint32_t*) data_ready, (uint32_t) layer, \
        counter.data_ptr<int>())
    bool i64 = ids.scalar_type() == at::kLong;
    TORCH_CHECK(i64 || ids.scalar_type() == at::kInt, "cpu_moe_issue: ids must be int32/int64");
    if (x.scalar_type() == at::kBFloat16) { if (i64) ISSUE(at::BFloat16, int64_t); else ISSUE(at::BFloat16, int32_t); }
    else if (x.scalar_type() == at::kHalf) { if (i64) ISSUE(at::Half, int64_t); else ISSUE(at::Half, int32_t); }
    else TORCH_CHECK(false, "cpu_moe_issue: x must be fp16/bf16");
    #undef ISSUE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cpu_moe_collect
(
    at::Tensor& out, int64_t sout, int64_t ho,
    int64_t ctrl, int64_t done, int64_t consumed, int64_t timeout_ns
)
{
    TORCH_CHECK(out.dim() == 2 && out.stride(1) == 1, "cpu_moe_collect: out layout");
    int rows = (int) out.size(0);
    if (rows == 0) return;
    const at::cuda::OptionalCUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #define COLLECT(TO) cpu_moe_collect_kernel<TO><<<rows, 256, 0, stream>>>( \
        (TO*) out.data_ptr(), out.stride(0), (const float*) sout, rows, (int) out.size(1), (int) ho, \
        (uint32_t*) ctrl, (const uint32_t*) done, (uint32_t*) consumed, (long long) timeout_ns)
    if (out.scalar_type() == at::kBFloat16) COLLECT(at::BFloat16);
    else if (out.scalar_type() == at::kHalf) COLLECT(at::Half);
    else TORCH_CHECK(false, "cpu_moe_collect: out must be fp16/bf16");
    #undef COLLECT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ngram_gather_dequant
(
    const at::Tensor& ids, int64_t table, int64_t num_rows, int64_t words, int64_t K,
    const at::Tensor& bias, at::Tensor& out
)
{
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kLong && ids.is_contiguous(), "ngram_gather_dequant: ids");
    TORCH_CHECK(bias.scalar_type() == at::kHalf && bias.is_contiguous() && bias.size(1) == ROW_DIM, "ngram_gather_dequant: bias");
    TORCH_CHECK(out.is_contiguous() && out.size(-1) == ROW_DIM && out.numel() == ids.numel() * ROW_DIM, "ngram_gather_dequant: out");
    TORCH_CHECK(words <= ROW_DIM && words == 1 + ROW_DIM * K / 16, "ngram_gather_dequant: words/K");
    int n = (int) ids.numel();
    if (n == 0) return;
    const at::cuda::OptionalCUDAGuard guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int heads = (int) bias.size(0);
    size_t smem = (size_t) words * sizeof(uint16_t);
    if (out.scalar_type() == at::kBFloat16)
        ngram_gather_dequant_kernel<at::BFloat16><<<n, ROW_DIM, smem, stream>>>(ids.data_ptr<int64_t>(),
            (const int16_t*) table, num_rows, (const __half*) bias.data_ptr(), (at::BFloat16*) out.data_ptr(), (int) K, (int) words, heads);
    else if (out.scalar_type() == at::kHalf)
        ngram_gather_dequant_kernel<at::Half><<<n, ROW_DIM, smem, stream>>>(ids.data_ptr<int64_t>(),
            (const int16_t*) table, num_rows, (const __half*) bias.data_ptr(), (at::Half*) out.data_ptr(), (int) K, (int) words, heads);
    else TORCH_CHECK(false, "ngram_gather_dequant: out dtype");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cpu_moe_issue", &cpu_moe_issue, "stage a routed-expert job for the CPU worker (graph-safe)");
    m.def("cpu_moe_collect", &cpu_moe_collect, "wait for the CPU worker and read its partial (graph-safe)");
    m.def("ngram_gather_dequant", &ngram_gather_dequant, "gather + decode EXL3 n-gram rows from host memory");
}
