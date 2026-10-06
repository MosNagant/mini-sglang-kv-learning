"""Complete, visible learning experiment; uses the existing CUDA kernel.

The coach supplies input preparation, observation and recording.
The learner adds correctness checks at the end. This is not a benchmark.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from minisgl.kernel import store_cache

# A. Prepare the destination and inputs on the GPU.
capacity, batch, width = 8, 2, 128
kv_cache = torch.full((capacity, 2, width), -1, dtype=torch.float16, device="cuda")
k_cache = kv_cache[:, 0, :]
v_cache = kv_cache[:, 1, :]

k = torch.arange(batch * width, dtype=torch.float16, device="cuda").reshape(batch, width) + 10
v = k + 1000
indices = torch.tensor([5, 1], dtype=torch.int32, device="cuda")

print("CACHE:", tuple(kv_cache.shape), kv_cache.dtype, kv_cache.device)
print("K VIEW:", tuple(k_cache.shape), "stride:", k_cache.stride())
print("INPUT K:", tuple(k.shape), "stride:", k.stride())
print("INDICES:", indices.cpu().tolist())
print("BEFORE: K/V cache values were initialized to -1.")

# B. Call mini-sglang's existing CUDA kernel, then wait for GPU completion.
store_cache(k_cache, v_cache, indices, k, v)
torch.cuda.synchronize()

# C. Display a small part of every destination row.
for row in range(capacity):
    print(f"CACHE {row}: K[:4]={k_cache[row, :4].cpu().tolist()} V[:4]={v_cache[row, :4].cpu().tolist()}")

# D. Coach's record keeping: preserve this source and ALL values before checks.
record_parent = Path(__file__).resolve().parent / "learning" / "complete-kv"
record_parent.mkdir(parents=True, exist_ok=True)
record_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=record_parent))
shutil.copyfile(__file__, record_dir / "kv_write_lab.py")
record = {
    "scope": "learning observation and learner checks; not a benchmark",
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "cache_initial_value": -1,
    "cache_shape": list(kv_cache.shape),
    "k_input_stride": list(k.stride()),
    "k_cache_stride": list(k_cache.stride()),
    "dtype": str(k.dtype),
    "indices": indices.cpu().tolist(),
    "k": k.cpu().tolist(),
    "v": v.cpu().tolist(),
    "k_cache": k_cache.cpu().tolist(),
    "v_cache": v_cache.cpu().tolist(),
}
(record_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("RAW_RESULT_SAVED:", record_dir / "result.json")

# E. Learner's check area. Write checks here after understanding A-C.
# Observation output alone does not establish correctness.
# Keep this area at the top level; no extra function definition is needed.
assert torch.equal(k_cache[5], k[0])
assert torch.equal(v_cache[5], v[0])
assert torch.equal(k_cache[1], k[1])
assert torch.equal(v_cache[1], v[1])
torch.all(x == -1).item()
print("RUN_FINISHED: review the saved code for check coverage; no full-suite or timing PASS is claimed.")
