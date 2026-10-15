"""M08 metadata-cache candidate, coach implementation, 2026-10-15.

The device kernel, launch grid, warp count and value contract come unchanged
from 40-m08-kv-backends.py. This candidate only caches successfully validated
metadata and its derived launch parameters. EVERY Python call reads a complete
signature of all five tensors: type, shape, strides, layout, dtype and device.
New signatures pass the original validator; failed validation is never cached.
The bounded cache stores no tensors, object IDs, addresses or CUDA streams.

Indices must still be unique and in range; source/index storage must not alias
writable cache elements, and K/V live cache regions must not overlap. Those
caller-owned value/alias conditions were not checked by the original adapter
and are not checked here. No device values are read, copied or synchronized.
Tensor metadata must not be mutated concurrently with a call.

Warm up all required signatures and JIT specializations before Graph capture.
Graph replay bypasses Python and requires fixed addresses/metadata and a valid
index contract on every replay. No performance improvement is claimed here.
Inject base_module when freezing a worker so both variants use the SAME loaded
40-module kernel object. With no injection, the sibling 40 file is loaded.
"""

from collections.abc import Callable
from functools import cache
import importlib.util
from pathlib import Path
import sys

AUTHORSHIP = "coach implementation; learner mastery not assessed"
MAX_SIGNATURES = 64


@cache
def _load_base():
    path = Path(__file__).resolve().with_name("40-m08-kv-backends.py")
    name = "_m08_metadata_cache_base_" + str(path).encode().hex()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _signature(torch, tensors):
    result = []
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        result.append((type(tensor), tuple(tensor.shape), tuple(tensor.stride()),
                       tensor.layout, tensor.dtype, tensor.device))
    return tuple(result)


def make_store_cache(backend: str, *, original_store: Callable | None = None,
                     observe: bool = False, base_module=None) -> Callable:
    """Expose the 40/41-compatible entry with bounded metadata memoization.

    Original delegates to the unchanged base factory. Triton always launches
    the unchanged base JIT kernel with current tensors and the current stream.
    metadata_cache_info() is for inspection OUTSIDE measured intervals; the
    non-observing hot path does not update custom diagnostic counters.
    """
    if backend not in ("original", "triton"):
        raise ValueError("backend must be explicitly 'original' or 'triton'")
    base = _load_base() if base_module is None else base_module
    if backend == "original":
        return base.make_store_cache("original", original_store=original_store, observe=observe)

    import torch
    triton, kernel = base._triton_kernel()
    # FIFO eviction avoids additional hit-side recency/counter updates.
    # Values contain only immutable scalar metadata, never Tensor objects.
    metadata_cache = {}

    def store_cache(k_cache, v_cache, indices, k, v):
        tensors = (("k_cache", k_cache), ("v_cache", v_cache),
                   ("indices", indices), ("k", k), ("v", v))
        signature = _signature(torch, tensors)
        metadata = metadata_cache.get(signature)
        if metadata is None:
            validated = base._validate_metadata(torch, k_cache, v_cache, indices, k, v)
            metadata = (*validated, triton.next_power_of_2(validated[2]))
            if len(metadata_cache) >= MAX_SIGNATURES:
                del metadata_cache[next(iter(metadata_cache))]
            metadata_cache[signature] = metadata
        b, c, width, input_k_stride, input_v_stride, cache_k_stride, cache_v_stride, block = metadata
        if b == 0:
            return

        compiled = kernel[(b,)](k_cache, v_cache, indices, k, v,
                                INPUT_K_STRIDE=input_k_stride, INPUT_V_STRIDE=input_v_stride,
                                CACHE_K_STRIDE=cache_k_stride, CACHE_V_STRIDE=cache_v_stride,
                                D=width, BLOCK_D=block, num_warps=base.NUM_WARPS)
        if observe:
            store_cache.compiled_kernel = compiled
            store_cache.call_count += 1
            store_cache.launch_count += 1
            store_cache.last_launch = {
                "backend": "triton", "B": b, "C": c, "D": width,
                "input_k_stride": input_k_stride, "input_v_stride": input_v_stride,
                "cache_k_stride": cache_k_stride, "cache_v_stride": cache_v_stride,
                "grid": [b], "BLOCK_D": block, "num_warps": base.NUM_WARPS,
                "counter_scope": "Python calls including capture; excludes Graph replay",
            }

    def metadata_cache_info():
        return {"entries": len(metadata_cache), "max_entries": MAX_SIGNATURES,
                "eviction": "FIFO", "custom_hit_miss_counters": False,
                "scope": "validated tensor metadata only; no tensor/pointer/stream caching"}

    store_cache.backend = backend
    store_cache.kernel = kernel
    store_cache.authorship = AUTHORSHIP
    store_cache.call_count = 0
    store_cache.launch_count = 0
    store_cache.last_launch = None
    store_cache.compiled_kernel = None
    store_cache.metadata_cache_info = metadata_cache_info
    return store_cache
