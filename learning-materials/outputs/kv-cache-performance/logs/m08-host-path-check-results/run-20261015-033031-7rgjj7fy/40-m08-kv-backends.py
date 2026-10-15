"""M08 explicit KV-store adapters -- coach implementation, 2026-10-14.

This is a new research adapter, not a change to the frozen M06/M07 kernels or
mini-sglang. It has no import-time CUDA initialization, allocation or compilation.
Create the selected callable before patching minisgl.kernel.store_cache:

    original = minisgl.kernel.store_cache
    store = make_store_cache("triton", original_store=original)

Both backends accept CUDA FP16 strided tensors shaped (rows, D) or
(rows, heads, head_dim), with contiguous trailing dimensions and positive,
non-overlapping rows. Each flattened row has 1 <= D <= 4096 elements. Triton
uses independent K/V input/cache row strides. Original additionally enforces
the fixed upstream CUDA kernel's common-stride and vector-alignment contract.

CALLER-OWNED VALUE CONTRACT: indices are unique, with 0 <= indices[i] < C.
Inputs/indices do not alias writable cache elements; K/V cache elements do not
overlap. Their lifetimes and CUDA stream ordering are the caller's responsibility.
No GPU values are checked here, including during CUDA Graph capture or replay.
Do diagnostic value checks separately, and warm up each signature before capture.
Graph replay does not rerun this Python metadata validator: tensor metadata and
addresses must stay fixed and the index contract must hold on every replay.

There is no implicit contiguous copy, conversion, backend fallback, autograd
support or performance claim. GPU/model verification is separate evidence.
"""

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

AUTHORSHIP = "coach implementation; learner mastery not assessed"
MAX_WIDTH = 4096
NUM_WARPS = 4


def _row_layout(tensor, name: str) -> tuple[int, int, int]:
    """Validate dense trailing dimensions using metadata only; never reshape."""
    if tensor.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape (rows, D) or (rows, H, d)")
    width = 1
    for axis in range(tensor.ndim - 1, 0, -1):
        size = tensor.shape[axis]
        if size <= 0 or tensor.stride(axis) != width:
            raise ValueError(f"{name} trailing dimensions must be nonempty and contiguous")
        width *= size
    row_stride = tensor.stride(0)
    if row_stride < width:
        raise ValueError(f"{name} rows must not overlap; stride(0) must be >= D")
    return tensor.shape[0], width, row_stride


def _validate_metadata(torch, k_cache, v_cache, indices, k, v):
    """No item(), CPU transfers, reductions, synchronization or allocations."""
    tensors = (("k_cache", k_cache), ("v_cache", v_cache),
               ("indices", indices), ("k", k), ("v", v))
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on CUDA")
        if tensor.device != k_cache.device:
            raise ValueError("all five tensors must be on the same CUDA device")
        expected_dtype = torch.int32 if name == "indices" else torch.float16
        if tensor.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype}")

    c, width, cache_k_stride = _row_layout(k_cache, "k_cache")
    vc, vd, cache_v_stride = _row_layout(v_cache, "v_cache")
    b, kd, input_k_stride = _row_layout(k, "k")
    vb, vd_input, input_v_stride = _row_layout(v, "v")
    if c != vc or b != vb or len({width, vd, kd, vd_input}) != 1:
        raise ValueError("K/V must agree on B, caches on C, and all flattened widths on D")
    if not 1 <= width <= MAX_WIDTH:
        raise ValueError(f"supported flattened width is 1 <= D <= {MAX_WIDTH}")
    if indices.ndim != 1 or indices.shape[0] != b or indices.stride(0) != 1:
        raise ValueError("indices must be a contiguous one-dimensional tensor of length B")
    if b > c:
        raise ValueError("unique valid destinations require B <= C")
    return b, c, width, input_k_stride, input_v_stride, cache_k_stride, cache_v_stride


def _validate_original_layout(width, strides, k_cache, v_cache, k, v):
    input_k_stride, input_v_stride, cache_k_stride, cache_v_stride = strides
    if input_k_stride != input_v_stride or cache_k_stride != cache_v_stride:
        raise ValueError("original CUDA requires equal K/V input strides and equal K/V cache strides")
    row_bytes = width * 2  # The shared contract has already enforced FP16.
    if row_bytes % 128:
        raise ValueError("original CUDA requires D * 2 to be a multiple of 128 bytes")
    alignment = 16 if row_bytes % 512 == 0 else (8 if row_bytes % 256 == 0 else 4)
    for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache), ("k", k), ("v", v)):
        if tensor.data_ptr() % alignment or tensor.stride(0) * 2 % alignment:
            raise ValueError(f"original CUDA requires {name} base/row alignment of {alignment} bytes")


@cache
def _triton_kernel():
    """Load JIT support only for the explicitly selected Triton backend."""
    global tl
    import triton
    import triton.language as tl

    # tl is bound in module globals so the JIT resolves it without closure
    # handling. Importing this module alone still needs neither Torch nor Triton.
    def m08_store_rows(K_CACHE, V_CACHE, INDICES, K, V,
                       INPUT_K_STRIDE: tl.constexpr, INPUT_V_STRIDE: tl.constexpr,
                       CACHE_K_STRIDE: tl.constexpr, CACHE_V_STRIDE: tl.constexpr,
                       D: tl.constexpr, BLOCK_D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        dst = tl.load(INDICES + row).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        kval = tl.load(K + row * INPUT_K_STRIDE + cols, mask=mask, other=0)
        vval = tl.load(V + row * INPUT_V_STRIDE + cols, mask=mask, other=0)
        tl.store(K_CACHE + dst * CACHE_K_STRIDE + cols, kval, mask=mask)
        tl.store(V_CACHE + dst * CACHE_V_STRIDE + cols, vval, mask=mask)

    return triton, triton.jit(m08_store_rows)


def make_store_cache(backend: str, *, original_store: Callable | None = None,
                     observe: bool = False) -> Callable:
    """Return an upstream-signature store callable; backend must be explicit.

    Pass the captured original callable before installing an external patch.
    With backend='original', omission imports the fixed original module entry.
    With backend='triton', original_store is never invoked. Construct this
    callable and warm it up outside Graph capture; capture has metadata checks
    but no device-to-host validation. B=0 returns after metadata validation.
    Optional observe stores successful nonempty Python dispatch counts,
    last launch metadata and the Triton CompiledKernel return value.
    These counters do NOT count CUDA Graph replays or prove replay correctness.
    """
    if backend not in ("original", "triton"):
        raise ValueError("backend must be explicitly 'original' or 'triton'")
    import torch

    kernel = None
    triton = None
    if backend == "original":
        if original_store is None:
            from minisgl.kernel.store import store_cache as original_store
        if not callable(original_store):
            raise TypeError("original_store must be callable")
    else:
        triton, kernel = _triton_kernel()

    def store_cache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                    indices: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        b, c, width, *strides = _validate_metadata(torch, k_cache, v_cache, indices, k, v)
        if backend == "original":
            _validate_original_layout(width, strides, k_cache, v_cache, k, v)
        if b == 0:
            return
        if backend == "original":
            # view() is guaranteed by the validated trailing dimensions and
            # raises rather than copying. The upstream kernel requires 2D K/V.
            original_store(k_cache=k_cache.view(c, width), v_cache=v_cache.view(c, width),
                           indices=indices, k=k.view(b, width), v=v.view(b, width))
        else:
            input_k_stride, input_v_stride, cache_k_stride, cache_v_stride = strides
            compiled = kernel[(b,)](k_cache, v_cache, indices, k, v,
                                    INPUT_K_STRIDE=input_k_stride, INPUT_V_STRIDE=input_v_stride,
                                    CACHE_K_STRIDE=cache_k_stride, CACHE_V_STRIDE=cache_v_stride,
                                    D=width, BLOCK_D=triton.next_power_of_2(width), num_warps=NUM_WARPS)
            if observe:
                store_cache.compiled_kernel = compiled
        if observe:
            store_cache.call_count += 1
            store_cache.launch_count += 1
            store_cache.last_launch = {
                "backend": backend, "B": b, "C": c, "D": width,
                "input_k_stride": strides[0], "input_v_stride": strides[1],
                "cache_k_stride": strides[2], "cache_v_stride": strides[3],
                "grid": [b] if backend == "triton" else None,
                "BLOCK_D": triton.next_power_of_2(width) if backend == "triton" else None,
                "num_warps": NUM_WARPS if backend == "triton" else None,
                "counter_scope": "Python calls including capture; excludes Graph replay",
            }

    store_cache.backend = backend
    store_cache.kernel = kernel
    store_cache.authorship = AUTHORSHIP
    store_cache.call_count = 0
    store_cache.launch_count = 0
    store_cache.last_launch = None
    store_cache.compiled_kernel = None
    return store_cache
