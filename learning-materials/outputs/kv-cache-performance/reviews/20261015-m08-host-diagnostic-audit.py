"""Offline audit of a saved 45 M08 host diagnostic; no Torch/GPU/WSL imports.

Usage: python this_file.py RUN_DIRECTORY [--output AUDIT_JSON]
Validate each frozen file against that run's plan, never the mutable current
45 script. Rebuild every persisted Graph/replacement scatter from saved inputs.
Eager checks and any unsaved replacement check remain runtime-reported evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same_array(left, right, label):
    require(left.shape == right.shape, label + ": shape mismatch")
    require(left.dtype == right.dtype, label + ": dtype mismatch")
    require(left.tobytes(order="C") == right.tobytes(order="C"),
            label + ": byte mismatch")


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual),
            label + ": not finite")
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11),
            f"{label}: {actual!r} != recomputed {expected!r}")


def audit_snapshot(path: Path, batch: int):
    keys = {"actual", "expected", "input_before", "input_after",
            "indices_before", "indices_after"}
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == keys, str(path) + ": unexpected NPZ schema")
        data = {key: archive[key] for key in keys}
    before = data["input_before"]
    indices = data["indices_before"]
    require(before.shape == (batch, 4096) and before.dtype == np.dtype("float16"),
            str(path) + ": wrong QKV backing shape/dtype")
    require(indices.shape == (batch,) and indices.dtype == np.dtype("int32"),
            str(path) + ": wrong index shape/dtype")
    require(np.isfinite(before).all(), str(path) + ": nonfinite input fixture")
    require(np.all((indices >= 0) & (indices < 513)), str(path) + ": invalid indices")
    require(np.unique(indices).size == batch, str(path) + ": duplicate indices")
    is_replacement = path.stem.endswith("-replacement")
    value_increment = 13 if is_replacement else 1
    index_increment = 3 if is_replacement else 1
    independently_built_input = ((np.arange(batch * 4096, dtype=np.int32) % 997)
                                .astype(np.float16).reshape(batch, 4096))
    independently_built_input += np.float16(value_increment)
    independently_built_indices = np.arange(batch, dtype=np.int32) + 7 + index_increment
    same_array(before, independently_built_input, str(path) + " independently rebuilt changed fixture")
    same_array(indices, independently_built_indices, str(path) + " independently rebuilt changed destinations")
    reconstructed = np.full((2, 513, 8, 128), -7, dtype=np.float16)
    reconstructed[0].reshape(513, 1024)[indices] = before[:, 2048:3072]
    reconstructed[1].reshape(513, 1024)[indices] = before[:, 3072:4096]
    same_array(data["actual"], reconstructed, str(path) + " independently rebuilt full cache")
    same_array(data["expected"], reconstructed, str(path) + " saved oracle")
    same_array(data["input_after"], before, str(path) + " complete read-only QKV backing")
    same_array(data["indices_after"], indices, str(path) + " read-only indices")
    return {"file": path.name, "sha256": sha(path), "stored_arrays": len(data),
            "stored_array_elements": sum(x.size for x in data.values()),
            "stored_array_bytes": sum(x.nbytes for x in data.values()),
            "input_values_and_destinations_rebuilt_from_fixture_recipe": True,
            "independent_full_cache_reconstruction": True,
            "readonly_qkv_and_indices_byte_exact": True}


def audit(run: Path):
    run = run.resolve()
    plan = read(run / "plan.json")
    summary = read(run / "summary.json")
    require(summary.get("status") == "HOST_DIAGNOSTIC_COMPLETE", "run did not complete")
    require(plan.get("rounds") == 3 and plan.get("host_intervals") == 30
            and plan.get("calls_per_interval") == 200, "unreviewed measurement plan")
    require(plan.get("arms") == ["old", "cached"], "unreviewed host arms")
    files = plan["files"]
    required = {"40-m08-kv-backends.py", "44-m08-kv-cached-metadata.py",
                "45-m08-host-path-diagnostic.py"}
    require(required.issubset(files), "missing frozen adapter/runner source")
    source_count = 0
    for name, digest in files.items():
        target = (run / name).resolve()
        require(target.is_relative_to(run), "source manifest path escapes run: " + name)
        require(target.is_file() and sha(target) == digest, "frozen source hash mismatch: " + name)
        source_count += 1
    require(any(name.startswith("source/") for name in files), "upstream source missing")
    require(len(summary.get("rounds", [])) == 3, "incomplete round summaries")

    snapshots, compiled, statistics_rows = [], [], []
    reported_eager = reported_graph = reported_replacements = 0
    persisted_replacements = 0
    source_report = {name: files[name] for name in sorted(required)}
    for round_id in (1, 2, 3):
        out = run / f"round-{round_id}"
        worker = read(out / "result.json")
        require(worker == summary["rounds"][round_id - 1], "summary/worker mismatch")
        require(worker.get("round") == round_id and
                worker.get("status") == "HOST_DIAGNOSTIC_AND_GPU_CORRECTNESS_PASS",
                "worker identity/status mismatch")
        cases = worker.get("cases", [])
        require(len(cases) == 2 and [x["B"] for x in cases] == [1, 128],
                "unexpected batch cases")
        expected_npz = set()
        for case in cases:
            batch = case["B"]
            require(case.get("D") == 1024 and case.get("C") == 513,
                    "wrong case configuration")
            checks = case["gpu_checks"]
            require(len(checks) == 3 and [x["arm"] for x in checks] ==
                    ["original", "old", "cached"], "wrong GPU check arms")
            for check in checks:
                require(all(check.get(field) is True for field in
                            ("eager_full_backing", "graph_changed_values_full_backing", "readonly_inputs")),
                        "runtime check did not pass")
                reported_eager += 1
                reported_graph += 1
                name = f"B{batch}-{check['arm']}.npz"
                expected_npz.add(name)
                record = audit_snapshot(out / name, batch)
                record.update(round=round_id, batch=batch, kind="changed-value Graph replay",
                              arm=check["arm"])
                snapshots.append(record)
            require(case.get("replacement_objects_checked") is True,
                    "replacement object runtime check absent")
            reported_replacements += 1
            replacement_path = out / f"B{batch}-replacement.npz"
            if replacement_path.exists():
                expected_npz.add(replacement_path.name)
                record = audit_snapshot(replacement_path, batch)
                record.update(round=round_id, batch=batch, kind="replacement tensor objects", arm="cached")
                snapshots.append(record)
                persisted_replacements += 1

            require(set(case["kernel_hashes"]) == {"old", "cached"}, "kernel arm hashes missing")
            require(case["kernel_hashes"]["old"] == case["kernel_hashes"]["cached"],
                    "old/new compiled artifact hashes differ")
            for arm in ("old", "cached"):
                require(set(case["kernel_hashes"][arm]) == {"ptx", "ttgir"}, "unexpected compiler schema")
                for kind in ("ptx", "ttgir"):
                    path = out / f"B{batch}-{arm}.{kind}"
                    # The runner hashes in-memory compiler text then saves it.
                    # Windows newline normalization is accounted for explicitly.
                    digest = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                    require(digest == case["kernel_hashes"][arm][kind], "compiled text hash mismatch")
                    compiled.append({"file": path.relative_to(run).as_posix(),
                                     "file_sha256": sha(path), "compiler_text_sha256": digest})
            medians = {}
            require(set(case["host_samples_us"]) == {"old", "cached"}, "host sample arms missing")
            for arm in ("old", "cached"):
                values = case["host_samples_us"][arm]
                require(len(values) == 30 and all(isinstance(v, (int, float)) and
                        math.isfinite(v) and v > 0 for v in values), "invalid host interval samples")
                medians[arm] = statistics.median(values)
                close(case["host_medians_us"][arm], medians[arm], "host median")
            change = (medians["cached"] / medians["old"] - 1) * 100
            close(case["host_change_pct"], change, "host change percent")
            statistics_rows.append({"round": round_id, "B": batch,
                                    "host_medians_us": medians, "host_change_pct": change,
                                    "intervals_per_arm": 30, "calls_per_interval": 200})
        actual_npz = {path.name for path in out.glob("*.npz")}
        require(actual_npz == expected_npz, "unreviewed/missing NPZ files in " + str(out))

    return {
        "status": "OFFLINE_HOST_DIAGNOSTIC_EVIDENCE_AUDIT_PASS",
        "run": str(run), "audit_script_sha256": sha(Path(__file__)),
        "plan_sha256": sha(run / "plan.json"), "summary_sha256": sha(run / "summary.json"),
        "method": "Local saved JSON, source, compiler text and NPZ only; no GPU run or current-source identity assumption.",
        "frozen_source_hashes_checked": source_count, "frozen_adapter_runner_hashes": source_report,
        "worker_results_checked": 3, "case_results_checked": 6,
        "host_interval_averages_checked": 360, "calls_represented_by_host_intervals": 72000,
        "runtime_reported_eager_checks": reported_eager,
        "runtime_reported_changed_value_graph_checks": reported_graph,
        "runtime_reported_replacement_checks": reported_replacements,
        "persisted_graph_snapshots_independently_rebuilt": 18,
        "persisted_replacement_snapshots_independently_rebuilt": persisted_replacements,
        "npz_files_checked": len(snapshots),
        "stored_arrays_checked": sum(x["stored_arrays"] for x in snapshots),
        "stored_array_elements_checked": sum(x["stored_array_elements"] for x in snapshots),
        "stored_array_bytes_checked": sum(x["stored_array_bytes"] for x in snapshots),
        "compiled_text_files_checked": len(compiled),
        "old_cached_ptx_and_ttgir_identical_within_each_case": True,
        "recomputed_host_statistics": statistics_rows, "snapshots": snapshots, "compiled_files": compiled,
        "limits": [
            "Host timings replace only launch with a no-op; not GPU kernel times or model speedup.",
            "Runtime Eager checks are reported, not reconstructed from separate Eager NPZ snapshots.",
            "Only persisted replacement snapshots are independently reconstructed; other replacement checks are runtime reports.",
            "Graph snapshots validate isolated KV replay with changed values, not full-model Graph logits.",
            "Matching compiler artifacts support unchanged device specialization; not a hardware-counter or causal bottleneck claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    output = args.output or args.run / "evidence-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "npz_files_checked": result["npz_files_checked"],
                      "host_interval_averages_checked": result["host_interval_averages_checked"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
