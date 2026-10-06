"""Observe the upstream test's interleaved K/V layout on a tiny CUDA tensor.

Run after sourcing the project's experiments/activate.sh in WSL.
This is a layout observation, not a store-kernel test or benchmark.
"""

import json
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch


def main():
    project = Path.home() / "projects" / "mini-sglang-kv"
    if not (project / "experiments" / "activate.sh").is_file():
        raise SystemExit("Project not found. Run this script inside your weltlaw WSL account.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE: keep this error; no packages were changed.")

    revision = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    # Same logical layout as the upstream test, reduced to two tokens and width four.
    kv_cache = torch.tensor(
        [
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            [[30, 31, 32, 33], [40, 41, 42, 43]],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]
    k_contiguous = k_cache.contiguous()
    v_contiguous = v_cache.contiguous()

    records = {}
    lines = []

    def emit(message):
        print(message, flush=True)
        lines.append(message)

    emit("SCOPE=GPU tensor layout only; no store kernel or timing measurements")
    emit(f"GPU={torch.cuda.get_device_name(0)}")
    emit(f"BASE_VALUES_IN_ORDER={kv_cache.flatten().cpu().tolist()}")
    emit(f"BYTES_PER_ELEMENT={kv_cache.element_size()}")

    for name, tensor in [
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("k_contiguous", k_contiguous),
        ("v_contiguous", v_contiguous),
    ]:
        record = {
            "values": tensor.cpu().tolist(),
            "shape": list(tensor.shape),
            "stride_elements": list(tensor.stride()),
            "row_start_gap_bytes": tensor[1].data_ptr() - tensor[0].data_ptr(),
            "is_contiguous": tensor.is_contiguous(),
            "shares_storage_with_kv_cache": (
                tensor.untyped_storage().data_ptr()
                == kv_cache.untyped_storage().data_ptr()
            ),
        }
        records[name] = record
        emit(f"\n{name}")
        emit(f"  values={record['values']}")
        emit(f"  shape={record['shape']}  stride_elements={record['stride_elements']}")
        emit(f"  row_start_gap_bytes={record['row_start_gap_bytes']}")
        emit(
            f"  contiguous={record['is_contiguous']}  "
            f"shares_base_storage={record['shares_storage_with_kv_cache']}"
        )

    parent = project / "experiments" / "learning" / "layout"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    shutil.copyfile(Path(__file__), run / "inspect_layout.py")
    result = {
        "scope": "tiny GPU layout observation; not KV correctness or performance evidence",
        "created_at": datetime.now().astimezone().isoformat(),
        "project_revision": revision,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "base_shape": list(kv_cache.shape),
        "base_values_in_order": kv_cache.flatten().cpu().tolist(),
        "bytes_per_element": kv_cache.element_size(),
        "tensors": records,
    }
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    emit(f"\nRUN_DIRECTORY={run}")
    emit("LAYOUT_OBSERVATION_COMPLETE")
    (run / "output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
