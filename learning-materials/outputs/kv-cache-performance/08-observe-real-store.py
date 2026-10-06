"""Coach-provided observation scaffolding; invokes the existing CUDA store.

No new kernel, timing loop or correctness suite is implemented here.
The learner interprets the saved inputs and outputs before adding checks.
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


def main():
    project = Path.home() / "projects" / "mini-sglang-kv"
    expected_revision = "9a91cfafe754aa85daee49998176275667eb58f2"
    revision = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != expected_revision:
        raise SystemExit(f"COMMIT_MISMATCH: {revision}; no checkout was changed")

    parent = project / "experiments" / "learning" / "real-store"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    shutil.copyfile(__file__, run / "observe_real_store.py")
    lines = []

    def emit(message):
        print(message, flush=True)
        lines.append(message)
        (run / "output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    emit(f"RUN_DIRECTORY={run}")
    emit("SCOPE=existing CUDA store observation; NOT a benchmark or a correctness suite")
    sources = [
        "python/minisgl/kernel/store.py",
        "python/minisgl/kernel/csrc/jit/store.cu",
        "python/minisgl/kernel/utils.py",
    ]
    hashes = {}
    for relative in sources:
        path = project / relative
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    diff = subprocess.check_output(
        ["git", "-C", str(project), "diff", "HEAD", "--", *sources], text=True
    )
    (run / "source.diff").write_text(diff, encoding="utf-8")
    if diff:
        raise SystemExit("SOURCE_EDITS_PRESENT: preserved source.diff; review edits before labeling this upstream")

    import torch
    from minisgl.kernel import store_cache

    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE: keep the error; no packages were changed")

    # Interleaved cache follows the existing test layout, at a tiny capacity.
    capacity, batch, width = 8, 2, 128
    kv_cache = torch.full((capacity, 2, width), -1, dtype=torch.float16, device="cuda")
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]
    k = torch.arange(batch * width, dtype=torch.float16, device="cuda").reshape(batch, width) + 10
    v = k + 1000
    indices = torch.tensor([5, 1], dtype=torch.int32, device="cuda")

    emit(f"GPU={torch.cuda.get_device_name(0)}")
    emit(f"K_CACHE shape={tuple(k_cache.shape)} stride={k_cache.stride()} dtype={k_cache.dtype} device={k_cache.device}")
    emit(f"K_INPUT shape={tuple(k.shape)} stride={k.stride()}")
    emit(f"BYTES_PER_SCALAR={k.element_size()}")
    emit(f"INDICES={indices.cpu().tolist()}")
    emit("BEFORE: all K/V cache entries are -1")
    for row in range(batch):
        emit(f"INPUT {row}: K[:4]={k[row, :4].cpu().tolist()} V[:4]={v[row, :4].cpu().tolist()}")
    emit("CALLING: store_cache(k_cache, v_cache, indices, k, v)")

    # This calls upstream Python -> FFI/JIT -> CUDA, not Python list assignment.
    store_cache(k_cache, v_cache, indices, k, v)
    torch.cuda.synchronize()

    k_after, v_after = k_cache.cpu().tolist(), v_cache.cpu().tolist()
    for row in range(capacity):
        emit(f"CACHE {row}: K[:4]={k_after[row][:4]} V[:4]={v_after[row][:4]}")
    record = {
        "scope": "one real CUDA store invocation; no timings or automated correctness verdict",
        "created_at": datetime.now().astimezone().isoformat(),
        "project_revision": revision,
        "source_sha256": hashes,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cache_shape": list(kv_cache.shape),
        "k_cache_stride_elements": list(k_cache.stride()),
        "k_input_stride_elements": list(k.stride()),
        "dtype": str(k.dtype),
        "bytes_per_scalar": k.element_size(),
        "cache_initial_value": -1,
        "indices": indices.cpu().tolist(),
        "k_input": k.cpu().tolist(),
        "v_input": v.cpu().tolist(),
        "k_cache_after": k_after,
        "v_cache_after": v_after,
    }
    (run / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    emit("REAL_STORE_OBSERVATION_COMPLETE: inspect the result; no correctness PASS is claimed")
    emit("Console shows 4 elements per row; result.json contains all 128 elements per row")


if __name__ == "__main__":
    main()
