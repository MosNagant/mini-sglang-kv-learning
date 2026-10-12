"""M07 grouped candidate: fill exactly THREE expressions marked TODO.

Coach supplies contract checks, 2-D address broadcasting and combined masks.
Learner supplies input row numbers, valid-row condition and program count.
Both this candidate and frozen v1 use four warps. Four rows per program does
not assert that each warp handles exactly one row; compiler mapping is separate.
This checked public entry is not a timing/Graph entry.
"""
import triton
import triton.language as tl
from kv_contract_support import validate_metadata, validate_index_values

ROWS_PER_PROGRAM = 4
NUM_WARPS = 4


@triton.jit
def store_kv_grouped(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr,
                     BATCH: tl.constexpr, WIDTH: tl.constexpr,
                     SRC_ROW_STRIDE: tl.constexpr, DST_ROW_STRIDE: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    row_mask = rows < BATCH

    # Guard the index read BEFORE building 2-D source/destination pointers.
    positions = tl.load(indices_ptr + rows, mask=row_mask, other=0)
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < WIDTH
    valid = row_mask[:, None] & col_mask[None, :]
    src_offsets = rows[:, None] * SRC_ROW_STRIDE + cols[None, :]
    dst_offsets = positions[:, None] * DST_ROW_STRIDE + cols[None, :]
    k_values = tl.load(k_ptr + src_offsets, mask=valid, other=0)
    tl.store(k_cache_ptr + dst_offsets, k_values, mask=valid)
    v_values = tl.load(v_ptr + src_offsets, mask=valid, other=0)
    tl.store(v_cache_ptr + dst_offsets, v_values, mask=valid)


def store_cache_triton(k_cache, v_cache, indices, k, v):
    batch, width, src_stride, dst_stride, block = validate_metadata(
        k_cache, v_cache, indices, k, v)
    if indices.shape != (batch,):
        raise ValueError("INDEX_COUNT: one destination is required per input row")
    if batch == 0:
        return
    validate_index_values(indices, batch, k_cache.shape[0])
    grid = (triton.cdiv(batch, ROWS_PER_PROGRAM),)
    store_kv_grouped[grid](
        k_cache, v_cache, indices, k, v,
        BATCH=batch, WIDTH=width, SRC_ROW_STRIDE=src_stride,
        DST_ROW_STRIDE=dst_stride, BLOCK_SIZE=block,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM, num_warps=NUM_WARPS,
    )
