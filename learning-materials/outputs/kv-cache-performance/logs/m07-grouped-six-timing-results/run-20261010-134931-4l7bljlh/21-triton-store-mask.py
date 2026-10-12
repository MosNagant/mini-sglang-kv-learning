"""M06-03 KV store with column masks; coach demonstration requested by learner.

Extends the reviewed v1 address formulas without changing their element units.
Lesson: FP16 B3, widths64/128/192, capacity8, unique int32 indices[7,0,3].
K/V inputs share their row stride; K/V cache views share their row stride.
Column strides are1. BLOCK_SIZE is logical column count, not thread count.
Actual GPU acceptance is recorded by21-check-kv-mask.sh, not this docstring.
"""

import triton
import triton.language as tl


@triton.jit
def store_kv_v1(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr,
                WIDTH: tl.constexpr, SRC_ROW_STRIDE: tl.constexpr,
                DST_ROW_STRIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pos = tl.load(indices_ptr + pid)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < WIDTH
    src_offsets = pid * SRC_ROW_STRIDE + cols
    dst_offsets = pos * DST_ROW_STRIDE + cols

    k_values = tl.load(k_ptr + src_offsets, mask=mask, other=0)
    tl.store(k_cache_ptr + dst_offsets, k_values, mask=mask)
    v_values = tl.load(v_ptr + src_offsets, mask=mask, other=0)
    tl.store(v_cache_ptr + dst_offsets, v_values, mask=mask)


if __name__ == "__main__":
    print("Kernel definition only. Use21-check-kv-mask.sh for the lesson checks.")
