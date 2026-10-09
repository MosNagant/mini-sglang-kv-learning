"""M06-01: configurable-width Triton v1 draft; no performance claim.

Checked lesson cases: FP16 K/V shape (3, WIDTH), capacity=8, interleaved
cache, unique GPU int32 indices [7, 0, 3]. WIDTH is 64 or 128.
The launcher reads WIDTH and row strides from the actual tensors.
SRC_ROW_STRIDE is shared by K/V inputs; DST_ROW_STRIDE is shared by
K/V cache views. Column strides must be 1. All offsets count elements.
Different K/V row strides and other layouts are not covered in this lesson.

Coach: parameter declarations, launch adapter, checks and recording.
Learner: replace pass with cols/src_offsets/dst_offsets, extending the
reviewed fixed-width kernel. The four reviewed read/write lines are kept.
The old triton_store_first.py and any width exercise remain unchanged.
"""

import triton
import triton.language as tl


@triton.jit
def store_kv_v1(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr,
                WIDTH: tl.constexpr, SRC_ROW_STRIDE: tl.constexpr,
                DST_ROW_STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    pos = tl.load(indices_ptr + pid)

    # LEARNER_WIDTH_BEGIN
    cols = tl.arange(0, WIDTH)

    src_offsets = pid * SRC_ROW_STRIDE + cols
    dst_offsets = pos * DST_ROW_STRIDE + cols
    # LEARNER_WIDTH_END

    k_values = tl.load(k_ptr + src_offsets)
    tl.store(k_cache_ptr + dst_offsets, k_values)
    v_values = tl.load(v_ptr + src_offsets)
    tl.store(v_cache_ptr + dst_offsets, v_values)


if __name__ == "__main__":
    print("Kernel definition only. Use 14-check-kv-widths.sh after filling the three lines.")
