"""Learner width-parameter exercise; not a general operator or benchmark.

Test contract: FP16, K/V shape (3, WIDTH) with stride (WIDTH, 1);
capacity=8 interleaved cache, each K/V view stride (2 * WIDTH, 1);
unique int32 indices [7, 0, 3]; WIDTH is 64 or 128; grid=(3,).

The coach adds the WIDTH parameter and test runner. The learner replaces
pass with three address lines, extending their reviewed 64-wide draft.
"""

import triton
import triton.language as tl


@triton.jit
def store_kv_width(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr,
                   WIDTH: tl.constexpr):
    pid = tl.program_id(0)
    pos = tl.load(indices_ptr + pid)

    # LEARNER_WIDTH_BEGIN
    # Replace pass with your three lines defining:
    # cols, src_offsets, dst_offsets. Use WIDTH instead of fixed 64/128.
    pass
    # LEARNER_WIDTH_END

    k_values = tl.load(k_ptr + src_offsets)
    tl.store(k_cache_ptr + dst_offsets, k_values)
    v_values = tl.load(v_ptr + src_offsets)
    tl.store(v_cache_ptr + dst_offsets, v_values)


if __name__ == "__main__":
    print("Kernel definition only. Use 14-check-kv-widths.sh after filling the three lines.")
