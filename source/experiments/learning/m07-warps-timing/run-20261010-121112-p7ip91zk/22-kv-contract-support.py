"""M06-04 coach support for a checked public KV-store entry; not a benchmark.

Supported metadata: same CUDA device; FP16 rank-2 K/V and cache views; int32
rank-1 contiguous indices; D=64/128/192, C>0, 0<=B<=C; unit column strides;
equal K/V source row strides and equal K/V cache row strides. Source row
stride is D or D+64; cache row stride is D or 2D, plus 256/512 for D=192.
These strides permit the selected separate/interleaved layouts and guards,
not arbitrary strided tensors. This module does not import or launch Triton.

The learner-owned outer entry checks len(indices)==B, handles B==0 after
metadata validation, and supplies the launch. Neither learner decision is
implemented here. Metadata checks use host metadata only. Index-value checks
copy indices to the CPU once and synchronize CUDA work; they belong to the
checked public entry, NEVER silently inside a future kernel-only timed path.

Alias policy checks actual live row byte intervals, not storage identity or
whole-storage bounding boxes. Disjoint K/V cache views sharing one allocation
are allowed. Cache live regions must not overlap each other, either source,
or the visible indices region. This is conservative across ALL cache rows,
even if a particular invocation writes only some rows. Source K/V may overlap
each other because both are read-only. Padding/guard bytes are not live regions.
The caller must not concurrently change indices or aliased storage during
validation/launch. CPU metadata tests do not establish CUDA/Triton correctness.
"""


def _row_intervals(tensor):
    """Sorted half-open live byte intervals for an already-validated 2-D view."""
    rows, width = tensor.shape
    if not rows:
        return
    item_size = tensor.element_size()
    start = tensor.data_ptr()  # Includes this view's storage offset.
    step = tensor.stride(0) * item_size
    row_bytes = width * item_size
    for row in range(rows):
        low = start + row * step
        yield low, low + row_bytes


def _index_intervals(indices):
    """The visible contiguous index view, including its own storage offset."""
    if indices.shape[0]:
        low = indices.data_ptr()
        yield low, low + indices.shape[0] * indices.element_size()


def _intervals_overlap(left, right):
    """Linear two-pointer intersection; adjacent interval endpoints are safe."""
    a = next(left, None)
    b = next(right, None)
    while a is not None and b is not None:
        if a[1] <= b[0]:
            a = next(left, None)
        elif b[1] <= a[0]:
            b = next(right, None)
        else:
            return True
    return False


def validate_metadata(k_cache, v_cache, indices, k, v):
    """Return (B, D, source_row_stride, cache_row_stride, block_size).

    Raises TypeError for non-Tensors or unsupported dtypes, and ValueError
    for unsupported metadata or write aliases. Does not inspect index values,
    check index length, launch a kernel, or turn an empty batch into a no-op.
    The outer entry owns length validation and the B==0 branch. Empty tensors
    still require the specified row/column/index strides (ordinary contiguous
    empty views satisfy these); no source data pointer is used when B==0.
    """
    import torch

    named = (("k_cache", k_cache), ("v_cache", v_cache),
             ("indices", indices), ("k", k), ("v", v))
    for name, tensor in named:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("TENSOR_TYPE: " + name + " must be a torch.Tensor")
    for name, tensor in named:
        expected_rank = 1 if name == "indices" else 2
        if tensor.ndim != expected_rank:
            raise ValueError("TENSOR_RANK: " + name)
        expected_dtype = torch.int32 if name == "indices" else torch.float16
        if tensor.dtype != expected_dtype:
            raise TypeError("TENSOR_DTYPE: " + name)
    device = k.device
    if any(tensor.device.type != "cuda" for _, tensor in named):
        raise ValueError("CUDA_DEVICE: all five tensors must be CUDA tensors")
    if any(tensor.device != device for _, tensor in named):
        raise ValueError("SAME_DEVICE: all five tensors must use one CUDA device")

    batch, width = k.shape
    if v.shape != k.shape:
        raise ValueError("INPUT_SHAPE: K/V must have the same (B,D)")
    if width not in (64, 128, 192):
        raise ValueError("WIDTH_SCOPE: D must be 64, 128, or 192")
    capacity, cache_width = k_cache.shape
    if v_cache.shape != k_cache.shape or cache_width != width:
        raise ValueError("CACHE_SHAPE: caches must have the same (C,D)")
    if capacity <= 0 or not 0 <= batch <= capacity:
        raise ValueError("BATCH_CAPACITY: require C>0 and 0<=B<=C")
    if indices.stride() != (1,):
        raise ValueError("INDEX_STRIDE: require contiguous rank-1 indices")
    for name, tensor in (("k", k), ("v", v),
                         ("k_cache", k_cache), ("v_cache", v_cache)):
        if tensor.stride(1) != 1:
            raise ValueError("COLUMN_STRIDE: " + name)
    src, dst = k.stride(0), k_cache.stride(0)
    if v.stride(0) != src:
        raise ValueError("SHARED_SOURCE_STRIDE: K/V row strides must match")
    if v_cache.stride(0) != dst:
        raise ValueError("SHARED_CACHE_STRIDE: cache row strides must match")
    if src not in (width, width + 64):
        raise ValueError("SOURCE_LAYOUT_SCOPE: require D or D+64")
    allowed_dst = (width, 2 * width, 256, 512) if width == 192 else (width, 2 * width)
    if dst not in allowed_dst:
        raise ValueError("CACHE_LAYOUT_SCOPE: unsupported cache row stride")

    if _intervals_overlap(_row_intervals(k_cache), _row_intervals(v_cache)):
        raise ValueError("CACHE_WRITE_ALIAS: K/V cache live regions overlap")
    for cache_name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        for source_name, source in (("k", k), ("v", v)):
            if _intervals_overlap(_row_intervals(cache), _row_intervals(source)):
                raise ValueError("SOURCE_CACHE_ALIAS: " + cache_name + "/" + source_name)
        if _intervals_overlap(_row_intervals(cache), _index_intervals(indices)):
            raise ValueError("INDEX_CACHE_ALIAS: " + cache_name)
    block = 1 << (width - 1).bit_length()
    return batch, width, src, dst, block


def validate_index_values(indices, batch, capacity):
    """Reject duplicate or out-of-range destinations before the checked launch.

    Preconditions: metadata and learner-owned length checks have passed, and
    0<=B<=C with C>0. The caller skips this function for B==0; the defensive
    empty return here also performs no transfer. For B>0, exactly one blocking
    indices.cpu().tolist() obtains a stable host snapshot for range/uniqueness
    checks. This transfer synchronizes CUDA work and adds O(B) host work; report
    public-entry timing separately from any future kernel-only benchmark.
    No length check is duplicated here. No tensor is mutated and no GPU kernel
    is launched by this module. Invalid indices must never reach the store.
    """
    if batch == 0:
        return
    values = indices.cpu().tolist()
    if any(position < 0 or position >= capacity for position in values):
        raise ValueError("INDEX_RANGE: require 0<=indices[i]<C")
    if len(set(values)) != len(values):
        raise ValueError("INDEX_DUPLICATE: destinations must be unique")
