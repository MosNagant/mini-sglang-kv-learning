"""M07 candidate public launcher; coach scaffold, learner changes one parameter.

Change only NUM_WARPS from 4 to 1 for the first experiment.
The accepted kernel, addresses, masks, grid and contract stay unchanged.
This public wrapper validates index values and may synchronize. It is NOT a
trusted timing/Graph entry. Use 26-check-kv-warps.sh for correctness only.
"""
from kv_contract_support import validate_metadata, validate_index_values
from triton_store_mask import store_kv_v1

# LEARNER_CHANGE: leave the frozen v1 files alone; change this 4 to 1.
NUM_WARPS = 1


def store_cache_triton(k_cache, v_cache, indices, k, v):
    batch, width, src_stride, dst_stride, block = validate_metadata(
        k_cache, v_cache, indices, k, v)
    if indices.shape != (batch,):
        raise ValueError("INDEX_COUNT: one destination is required per input row")
    if batch == 0:
        return
    validate_index_values(indices, batch, k_cache.shape[0])
    store_kv_v1[(batch,)](
        k_cache, v_cache, indices, k, v,
        WIDTH=width, SRC_ROW_STRIDE=src_stride,
        DST_ROW_STRIDE=dst_stride, BLOCK_SIZE=block, num_warps=NUM_WARPS,
    )
