// Device-side expert cache manager for the pointer-table MoE path (kernels/offload_moe.py ExpertCache).
//
// One global pool of S GPU slots shared by all layers (slot = one expert record). State, all on the device:
//   slot_of[L * E]  int32   slot of global expert g = layer * E + e, -1 = host-resident
//   owner[S]        int32   global expert in slot s, -1 = free
//   stamp[S]        int64   step clock of the last use of slot s (hit or admission)
//   ref[S]          int32   CLOCK reference bit
//   hand[1]         int32   CLOCK hand;  clock[1] int64 step clock (one tick per cache_step call)
//   tables[L, E, F] int64   the per-layer pointer tables the GEMMs read (row = record field addresses)
//   admit[L, E, F]  int64   admission rows consumed by the fused write-back GEMM (0 = none)
//   stats[L, 2]     int64   distinct routed experts that hit / missed, per layer
// cache_step(layer, topk_ids) runs BEFORE the layer's MoE: dedups the routed ids, stamps hits, picks a victim per miss
// with CLOCK (never a slot used in this step), evicts the victim's previous owner (its table row goes back to the host
// bank, possibly of another layer), and writes the miss's admission row (destination = the victim slot). The miss's
// table row keeps pointing at the host bank for this step: the GEMM reads it zero-copy and writes it into the slot.
// cache_commit(layer, topk_ids) runs AFTER the MoE: copies consumed admission rows into the table and clears them.
// Both are single-block kernels with static shapes: CUDA-graph capturable. Misses beyond what the pool can take in one
// step (all slots used in this step) are served zero-copy without admission.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define TRELLIS_CACHE_MAX_E 1024
#define TRELLIS_CACHE_MAX_F 8

__global__ __launch_bounds__(1024) void trellis_cache_step_kernel(
    const int* __restrict__ ids, int nids, int layer, int E, int S, int F, int* __restrict__ slot_of,
    int* __restrict__ owner, int64_t* __restrict__ stamp, int* __restrict__ ref, int* __restrict__ hand,
    int64_t* __restrict__ clock, int64_t* __restrict__ tables, int64_t* __restrict__ admit,
    const int64_t* __restrict__ host_bases, int64_t arena_base, int64_t rec, const int64_t* __restrict__ offs,
    int64_t* __restrict__ stats, int do_admit) {
  __shared__ int seen[TRELLIS_CACHE_MAX_E];
  __shared__ int miss_list[TRELLIS_CACHE_MAX_E];
  __shared__ int n_miss, n_hit;
  __shared__ int64_t c;
  __shared__ int64_t off_sh[TRELLIS_CACHE_MAX_F];
  const int t = threadIdx.x;
  for (int e = t; e < E; e += blockDim.x) seen[e] = 0;
  if (t < F) off_sh[t] = offs[t];
  if (t == 0) {
    n_miss = 0;
    n_hit = 0;
    c = ++clock[0];
  }
  __syncthreads();
  for (int s = t; s < nids; s += blockDim.x) {
    const int e = ids[s];
    if (e < 0 || e >= E) continue;  // dropped slot (sentinel E)
    if (atomicExch(&seen[e], 1) != 0) continue;
    const int g = layer * E + e;
    const int sl = slot_of[g];
    if (sl >= 0) {
      stamp[sl] = c;  // pinned for this step
      ref[sl] = 1;
      atomicAdd(&n_hit, 1);
    } else {
      miss_list[atomicAdd(&n_miss, 1)] = e;
    }
  }
  __syncthreads();
  if (t != 0) return;
  stats[2 * layer + 0] += n_hit;
  stats[2 * layer + 1] += n_miss;
  if (!do_admit || S == 0) return;
  int h = hand[0];
  for (int m = 0; m < n_miss; m++) {
    const int e = miss_list[m];
    int victim = -1;
    for (int tries = 0; tries < 2 * S + 1; tries++) {
      const int v = h;
      h = (h + 1 == S) ? 0 : h + 1;
      if (owner[v] < 0) { victim = v; break; }
      if (stamp[v] == c) continue;  // used in this step: never evicted
      if (ref[v]) { ref[v] = 0; continue; }
      victim = v;
      break;
    }
    if (victim < 0) break;  // every slot is in use this step: remaining misses stay zero-copy only
    const int o = owner[victim];
    if (o >= 0) {  // evict: the old owner goes back to its host bank row
      slot_of[o] = -1;
      const int ol = o / E, oe = o % E;
      int64_t* row = tables + (int64_t)o * F;
      const int64_t hb = host_bases[ol] + (int64_t)oe * rec;
      for (int f = 0; f < F; f++) row[f] = hb + off_sh[f];
    }
    const int g = layer * E + e;
    owner[victim] = g;
    slot_of[g] = victim;
    stamp[victim] = c;
    ref[victim] = 1;
    int64_t* arow = admit + (int64_t)e * F;
    const int64_t db = arena_base + (int64_t)victim * rec;
    for (int f = 0; f < F; f++) arow[f] = db + off_sh[f];
  }
  hand[0] = h;
}

__global__ __launch_bounds__(1024) void trellis_cache_commit_kernel(const int* __restrict__ ids, int nids, int E, int F,
                                                                    int64_t* __restrict__ table,
                                                                    int64_t* __restrict__ admit) {
  __shared__ int seen[TRELLIS_CACHE_MAX_E];
  const int t = threadIdx.x;
  for (int e = t; e < E; e += blockDim.x) seen[e] = 0;
  __syncthreads();
  for (int s = t; s < nids; s += blockDim.x) {
    const int e = ids[s];
    if (e < 0 || e >= E) continue;
    if (atomicExch(&seen[e], 1) != 0) continue;
    int64_t* arow = admit + (int64_t)e * F;
    if (arow[0] == 0) continue;
    int64_t* trow = table + (int64_t)e * F;
    for (int f = 0; f < F; f++) {
      trow[f] = arow[f];
      arow[f] = 0;
    }
  }
}

static void check_i32(const at::Tensor& t, int64_t n, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == at::kInt && t.is_contiguous() && (n < 0 || t.numel() == n), name, ": int32 CUDA");
}
static void check_i64(const at::Tensor& t, int64_t n, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == at::kLong && t.is_contiguous() && (n < 0 || t.numel() == n), name, ": int64 CUDA");
}

// tables / admit: int64 [L, E, F]; host_bases int64 [L]; offs int64 [F]; stats int64 [L, 2]
void moe_cache_step(const at::Tensor& ids, int64_t layer, at::Tensor& slot_of, at::Tensor& owner, at::Tensor& stamp,
                    at::Tensor& ref, at::Tensor& hand, at::Tensor& clock, at::Tensor& tables, at::Tensor& admit,
                    const at::Tensor& host_bases, int64_t arena_base, int64_t rec, const at::Tensor& offs,
                    at::Tensor& stats, bool do_admit) {
  const at::cuda::OptionalCUDAGuard device_guard(ids.device());
  TORCH_CHECK(tables.dim() == 3 && admit.sizes() == tables.sizes());
  const int64_t L = tables.size(0), E = tables.size(1), F = tables.size(2), S = owner.numel();
  TORCH_CHECK(E <= TRELLIS_CACHE_MAX_E && F <= TRELLIS_CACHE_MAX_F && layer >= 0 && layer < L);
  check_i32(ids, -1, "ids"); check_i32(slot_of, L * E, "slot_of"); check_i32(owner, S, "owner");
  check_i64(stamp, S, "stamp"); check_i32(ref, S, "ref"); check_i32(hand, 1, "hand"); check_i64(clock, 1, "clock");
  check_i64(tables, -1, "tables"); check_i64(admit, -1, "admit"); check_i64(host_bases, L, "host_bases");
  check_i64(offs, F, "offs"); check_i64(stats, 2 * L, "stats");
  trellis_cache_step_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      ids.data_ptr<int>(), (int)ids.numel(), (int)layer, (int)E, (int)S, (int)F, slot_of.data_ptr<int>(),
      owner.data_ptr<int>(), stamp.data_ptr<int64_t>(), ref.data_ptr<int>(), hand.data_ptr<int>(),
      clock.data_ptr<int64_t>(), tables.data_ptr<int64_t>(), admit[layer].data_ptr<int64_t>(),
      host_bases.data_ptr<int64_t>(), arena_base, rec, offs.data_ptr<int64_t>(), stats.data_ptr<int64_t>(),
      do_admit ? 1 : 0);
}

void moe_cache_commit(const at::Tensor& ids, int64_t layer, at::Tensor& tables, at::Tensor& admit) {
  const at::cuda::OptionalCUDAGuard device_guard(ids.device());
  TORCH_CHECK(tables.dim() == 3 && admit.sizes() == tables.sizes());
  const int64_t E = tables.size(1), F = tables.size(2);
  check_i32(ids, -1, "ids"); check_i64(tables, -1, "tables"); check_i64(admit, -1, "admit");
  TORCH_CHECK(E <= TRELLIS_CACHE_MAX_E && layer >= 0 && layer < tables.size(0));
  trellis_cache_commit_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      ids.data_ptr<int>(), (int)ids.numel(), (int)E, (int)F, tables[layer].data_ptr<int64_t>(),
      admit[layer].data_ptr<int64_t>());
}
