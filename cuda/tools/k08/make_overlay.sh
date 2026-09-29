#!/bin/bash
# copy of the integration tree (a6ac00e+) with the kernel lane's K07/K08 MoE extension + runtime overlaid (their tree untouched)
set -e
cd ~/freetoken-exl3
O=kernels/cuda_sm86/k08/ts_overlay
rm -rf $O; mkdir -p $O
rsync -a --exclude .git integration/trellis-serve/ $O/
K=kernels/cuda_sm86/trellis-serve/cuda
cp $K/csrc/build/lib/trellis_exl3_moe_kernels*.so $O/cuda/csrc/build/lib/
for f in kernels/offload_runtime.py kernels/offload_moe.py; do cp $K/src/sglang_exl3/$f $O/cuda/src/sglang_exl3/$f; done
diff <(cd integration/trellis-serve/cuda/src/sglang_exl3 && md5sum kernels/offload_store.py kernels/marlin_moe.py sglang_glue/offload_moe_method.py) \
     <(cd $K/src/sglang_exl3 && md5sum kernels/offload_store.py kernels/marlin_moe.py sglang_glue/offload_moe_method.py) && echo "other offload files identical"
echo overlay ready: $O
