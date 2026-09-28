// One MoE instantiation unit: TRELLIS_INST_MB=<0..4> (moe block family), TRELLIS_INST_CB=<0..2>, TRELLIS_INST_BITS=<3|4> (see setup.py).
#define MARLIN_NAMESPACE_NAME trellis_exl3_marlin_moe
#define TRELLIS_MOE_KERNEL_DEFINED
#include "exl3_marlin_moe_template.h"
#include "exl3_marlin_moe_kernels.h"

namespace MARLIN_NAMESPACE_NAME {
#define TRELLIS_MOE_INST(threads, tn, tk, mb, cb, bits) \
  template __global__ void TRELLIS_MOE_KERNEL(threads, tn, tk, mb, cb, bits)(TRELLIS_MOE_KERNEL_PARAMS);
TRELLIS_MOE_THREAD_CFGS(TRELLIS_MOE_INST, TRELLIS_INST_MB, TRELLIS_INST_CB, TRELLIS_INST_BITS)
}  // namespace MARLIN_NAMESPACE_NAME
