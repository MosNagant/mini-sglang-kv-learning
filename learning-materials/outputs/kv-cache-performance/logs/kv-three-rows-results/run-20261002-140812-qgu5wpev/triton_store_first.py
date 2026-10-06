"""First Triton KV store: one fixed learning case, not a general operator.

Contract: FP16 k/v shape (2, 128), stride (128, 1); cache K/V views
shape (8, 128), stride (256, 1); unique int32 indices [5, 1].
Launch exactly two programs. Columns 0..127 are all valid in this case.
The coach supplies the previously discussed address formulas and harness.
The learner writes the four read/write lines below. No timing is measured.
"""

import triton
import triton.language as tl


@triton.jit
def store_kv_first(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr):
    pid = tl.program_id(0)
    cols = tl.arange(0,64)
    pos = tl.load(indices_ptr + pid)

    src_offsets = pid * 64 + cols
    dst_offsets = pos * 128 + cols

    k_values = tl.load(k_ptr + src_offsets)
    tl.store(k_cache_ptr + dst_offsets, k_values)

    v_values = tl.load(v_ptr + src_offsets)
    tl.store(v_cache_ptr + dst_offsets, v_values)


if __name__ == "__main__":
    print("This file defines the kernel. Use 11-triton-store.sh to run its checks.")
