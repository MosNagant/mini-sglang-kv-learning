"""Seven bounded fixed-shape Gate/Up candidates; coach implementation.

Contract: x is contiguous CUDA FP16 (1,1024), weight is contiguous CUDA FP16
(6144,1024), both on the same device. Weights are ordered [3072 gate; 3072 up].
The first four candidates return the FP16 linear output (1,6144). The last
three return FP16 SiLU(gate)*up (1,3072), and therefore require a DIFFERENT
reference: F.linear followed by the project's silu_and_mul implementation.

Products/reductions are FP32; no quantization, TF32 or atomics are used. Fused
candidates explicitly round both linear intermediates to FP16 before computing
SiLU and the gate/up product in FP32, then round the output to FP16. Reduction
order and exponential implementation can differ from the reference: correctness
requires the separately frozen numerical contract, not assumed bitwise equality.

prepare() allocates output/workspace and performs optional weight packing once,
outside capture/timing. It synchronizes setup to make setup_seconds meaningful;
that field excludes JIT compilation and measures host-observed allocation/packing
setup only. call() performs no allocation, value check or synchronization and
returns the SAME reused output tensor. Current stream semantics are preserved.

The closure intentionally retains initial x, weight and output storage. call()
uses initial x; call_with_input(new_x) validates and passes the CURRENT input
tensor instead, allowing a model layer to reuse its prepared output and weights
without accidentally reading an earlier forward's input. No input may overlap
the plan's writable output/workspace. Weights are immutable for the plan's
lifetime; packed weights are a snapshot, so rebuild the plan if weights change.
Graph replay uses captured addresses and does not rerun this Python input guard.
Keep captured storage alive and update its values with correct stream ordering.
Do not overlap calls sharing a plan's output/workspace on different streams.
Inputs and original weights are never written. No performance claim is made.
"""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

AUTHORSHIP = "coach implementation; learner mastery not assessed"
K = 1024
N = 6144
HALF_N = 3072

_SPECS = (
    {"name": "gemv_direct_r4_w4", "kind": "gemv", "family": "direct", "row_tile": 4, "num_warps": 4,
     "description": "Direct FP32 dot reduction; four output rows per program, four warps."},
    {"name": "gemv_direct_r8_w4", "kind": "gemv", "family": "direct", "row_tile": 8, "num_warps": 4,
     "description": "Direct FP32 dot reduction; eight output rows per program, four warps."},
    {"name": "gemv_direct_r8_w8", "kind": "gemv", "family": "direct", "row_tile": 8, "num_warps": 8,
     "description": "Direct FP32 dot reduction; eight output rows per program, eight warps."},
    {"name": "gemv_splitk4_r4_w4", "kind": "gemv", "family": "split_k", "row_tile": 4, "num_warps": 4,
     "description": "Four K partitions with FP32 partial storage and a fixed second reduction; no atomics."},
    {"name": "swiglu_fused_r4_w4", "kind": "swiglu", "family": "fused", "row_tile": 4, "num_warps": 4,
     "description": "Four gate/up pairs per program, explicit FP16 intermediate rounding, fused SiLU multiply."},
    {"name": "swiglu_fused_r8_w4", "kind": "swiglu", "family": "fused", "row_tile": 8, "num_warps": 4,
     "description": "Eight gate/up pairs per program, explicit FP16 intermediate rounding, fused SiLU multiply."},
    {"name": "swiglu_packed_r4_w4", "kind": "swiglu", "family": "packed_fused", "row_tile": 4, "num_warps": 4,
     "description": "Four pairs per program with an actual contiguous [3072,2,1024] gate/up weight copy."},
)


def candidates():
    """Return independent descriptors in the predeclared experiment order."""
    return [dict(spec) for spec in _SPECS]


@triton.jit
def _gemv_direct(X, W, Y, TILE_N: tl.constexpr, K_SIZE: tl.constexpr, N_SIZE: tl.constexpr):
    rows = tl.program_id(0) * TILE_N + tl.arange(0, TILE_N)
    cols = tl.arange(0, K_SIZE)
    xv = tl.load(X + cols).to(tl.float32)
    wv = tl.load(W + rows[:, None] * K_SIZE + cols[None, :],
                 mask=rows[:, None] < N_SIZE, other=0).to(tl.float32)
    sums = tl.sum(wv * xv[None, :], axis=1)
    tl.store(Y + rows, sums.to(tl.float16), mask=rows < N_SIZE)


@triton.jit
def _gemv_splitk_partials(X, W, PARTIAL, TILE_N: tl.constexpr,
                         K_SIZE: tl.constexpr, N_SIZE: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * TILE_N + tl.arange(0, TILE_N)
    part = tl.program_id(1)
    cols = part * BLOCK_K + tl.arange(0, BLOCK_K)
    xv = tl.load(X + cols).to(tl.float32)
    wv = tl.load(W + rows[:, None] * K_SIZE + cols[None, :],
                 mask=rows[:, None] < N_SIZE, other=0).to(tl.float32)
    sums = tl.sum(wv * xv[None, :], axis=1)
    tl.store(PARTIAL + part * N_SIZE + rows, sums, mask=rows < N_SIZE)


@triton.jit
def _gemv_splitk_reduce(PARTIAL, Y, N_SIZE: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    parts = tl.arange(0, 4)
    values = tl.load(PARTIAL + parts[:, None] * N_SIZE + rows[None, :],
                     mask=rows[None, :] < N_SIZE, other=0)
    sums = tl.sum(values, axis=0)
    tl.store(Y + rows, sums.to(tl.float16), mask=rows < N_SIZE)


@triton.jit
def _swiglu_fused(X, W, Y, TILE_N: tl.constexpr, K_SIZE: tl.constexpr,
                  PAIRS: tl.constexpr, PACKED: tl.constexpr):
    rows = tl.program_id(0) * TILE_N + tl.arange(0, TILE_N)
    cols = tl.arange(0, K_SIZE)
    xv = tl.load(X + cols).to(tl.float32)
    if PACKED:
        gate_offsets = (2 * rows[:, None]) * K_SIZE + cols[None, :]
        up_offsets = (2 * rows[:, None] + 1) * K_SIZE + cols[None, :]
    else:
        gate_offsets = rows[:, None] * K_SIZE + cols[None, :]
        up_offsets = (rows[:, None] + PAIRS) * K_SIZE + cols[None, :]
    gates = tl.load(W + gate_offsets, mask=rows[:, None] < PAIRS, other=0).to(tl.float32)
    ups = tl.load(W + up_offsets, mask=rows[:, None] < PAIRS, other=0).to(tl.float32)
    gate_sum = tl.sum(gates * xv[None, :], axis=1)
    up_sum = tl.sum(ups * xv[None, :], axis=1)
    # Match the original two-operator boundary: linear outputs are FP16.
    gate = gate_sum.to(tl.float16).to(tl.float32)
    up = up_sum.to(tl.float16).to(tl.float32)
    activated = gate / (1.0 + tl.exp(-gate))
    result = activated * up
    tl.store(Y + rows, result.to(tl.float16), mask=rows < PAIRS)


def _validate(x, weight):
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be torch.Tensor objects")
    if x.shape != (1, K) or weight.shape != (N, K):
        raise ValueError("Fixed scope requires x(1,1024), weight(6144,1024)")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("Both x and weight must be FP16")
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("Both tensors must be on the same CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Both tensors must be contiguous")
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError("Both tensors must have strided layout")


def prepare(name, x, weight):
    """Prepare one fixed-address callable; all setup is outside measured calls.

    compiled_kernels is filled by actual calls and can be inspected afterwards.
    The references are never a claim that compilation or execution has occurred
    before call(). A split-K plan launches two kernels; all other plans launch one.
    """
    spec = next((dict(entry) for entry in _SPECS if entry["name"] == name), None)
    if spec is None:
        raise ValueError(f"Unknown frozen candidate: {name}")
    _validate(x, weight)
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare() must run outside CUDA Graph capture")
        torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        output = torch.empty((1, N if spec["kind"] == "gemv" else HALF_N), dtype=torch.float16, device=x.device)
        partial = None
        packed = None
        if spec["family"] == "split_k":
            partial = torch.empty((4, N), dtype=torch.float32, device=x.device)
        if spec["family"] == "packed_fused":
            packed = torch.empty((HALF_N, 2, K), dtype=torch.float16, device=x.device)
            packed.copy_(weight.view(2, HALF_N, K).permute(1, 0, 2))
        torch.cuda.synchronize(x.device)
        setup_seconds = time.perf_counter() - started

    tile, warps, family = spec["row_tile"], spec["num_warps"], spec["family"]
    compiled = [None, None] if partial is not None else [None]
    kernel_weight = weight if packed is None else packed
    device = x.device

    def call_with_input(new_x):
        if not isinstance(new_x, torch.Tensor):
            raise TypeError("Current input must be a torch.Tensor")
        if (new_x.shape != (1, K) or new_x.dtype != torch.float16 or
                new_x.device != device or new_x.layout != torch.strided or
                not new_x.is_contiguous()):
            raise ValueError("Current input must be contiguous FP16 (1,1024) on the prepared CUDA device")
        if family == "direct":
            compiled[0] = _gemv_direct[(triton.cdiv(N, tile),)](
                new_x, weight, output, TILE_N=tile, K_SIZE=K, N_SIZE=N, num_warps=warps)
        elif family == "split_k":
            compiled[0] = _gemv_splitk_partials[(triton.cdiv(N, tile), 4)](
                new_x, weight, partial, TILE_N=tile, K_SIZE=K, N_SIZE=N,
                BLOCK_K=K // 4, num_warps=warps)
            compiled[1] = _gemv_splitk_reduce[(triton.cdiv(N, 256),)](
                partial, output, N_SIZE=N, BLOCK_N=256, num_warps=4)
        else:
            compiled[0] = _swiglu_fused[(triton.cdiv(HALF_N, tile),)](
                new_x, kernel_weight, output, TILE_N=tile, K_SIZE=K, PAIRS=HALF_N,
                PACKED=packed is not None, num_warps=warps)
        return output

    def call():
        return call_with_input(x)

    return {
        "name": name, "call": call, "call_with_input": call_with_input,
        "kind": spec["kind"], "output": output,
        "workspace_bytes": 0 if partial is None else partial.numel() * partial.element_size(),
        "prepack_bytes": 0 if packed is None else packed.numel() * packed.element_size(),
        "setup_seconds": setup_seconds,
        "setup_scope": "synchronized output/workspace allocation and optional packing; excludes JIT compilation",
        "description": spec["description"], "spec": spec,
        "compiled_kernels": compiled, "kernel_launches_per_call": len(compiled),
        "workspace": partial, "packed_weight": packed,
        "authorship": AUTHORSHIP,
        "lifetime_contract": "immutable weights; current x validated on every call; input cannot alias output/workspace; captured addresses kept alive; stream-ordered updates; no concurrent plan calls",
    }
