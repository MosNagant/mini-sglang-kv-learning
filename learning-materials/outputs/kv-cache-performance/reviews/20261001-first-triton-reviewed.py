"""Review reference: coach template plus the learner's submitted read/write body.

Correction: dst_offset -> dst_offsets; standard indentation/spacing.
Fixed case only: FP16 inputs (2,128) stride (128,1), cache views (8,128)
stride (256,1), unique int32 indices [5,1], launch grid (2,).
This reference is not the WSL student's saved file and has not run on the GPU.
"""

import triton
import triton.language as tl


@triton.jit
def store_kv_first(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr):
    pid = tl.program_id(0)
    cols = tl.arange(0, 128)
    pos = tl.load(indices_ptr + pid)

    src_offsets = pid * 128 + cols
    dst_offsets = pos * 256 + cols

    k_values = tl.load(k_ptr + src_offsets)
    tl.store(k_cache_ptr + dst_offsets, k_values)
    v_values = tl.load(v_ptr + src_offsets)
    tl.store(v_cache_ptr + dst_offsets, v_values)


if __name__ == "__main__":
    print("This file defines the kernel. Use 11-triton-store.sh to run its checks.")
