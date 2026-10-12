"""M06-04 learner launcher: fill the THREE ellipses below.

Coach supplies validation helpers and recording. Learner completes index count,
empty-input condition and grid using the contract discussed in class.
This checked public call may synchronize to validate GPU index VALUES.
It is NOT the future pure-kernel/Graph timing entry.
The accepted triton_store_mask.py kernel is imported, never rewritten here.
"""
from kv_contract_support import validate_metadata, validate_index_values
from triton_store_mask import store_kv_v1


def store_cache_triton(k_cache, v_cache, indices, k, v):
    batch, width, src_stride, dst_stride, block = validate_metadata(
        k_cache, v_cache, indices, k, v)

    # LEARNER_BEGIN: replace the three ... expressions, not the whole function.
    # One destination per input row (shape is a one-element tuple).
    if indices.shape != (batch,):
        raise ValueError("INDEX_COUNT: one destination is required per input row")

    # No input rows: return before reading index values or launching a kernel.
    if batch == 0:
        return

    validate_index_values(indices, batch, k_cache.shape[0])

    # Our current kernel assigns ONE program to ONE input K/V row.
    store_kv_v1[(batch,)](
        k_cache, v_cache, indices, k, v,
        WIDTH=width, SRC_ROW_STRIDE=src_stride,
        DST_ROW_STRIDE=dst_stride, BLOCK_SIZE=block, num_warps=4,
    )
    # LEARNER_END


if __name__ == "__main__":
    print("Fill THREE ellipses, then use 22-check-kv-wrapper.sh; no GPU launched here.")
