"""Independent CPU audit of frozen 41 artifacts; imports no GPU/model code."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def audit(run):
    request, result = read(run / "request.json"), read(run / "result.json")
    assert result["status"] == "MODEL_AB_CORRECTNESS_PASS"
    assert request["engine_options"]["cuda_graph_bs"] == []
    assert request["engine_options"]["cache_type"] == "naive"
    assert len(request["input_token_ids"]) == 128 and request["max_new_tokens"] == 32
    source = read(run / "source-manifest.json")
    for name, expected in source.items():
        assert sha(run / "source" / name) == expected, name
    source_extra = {
        "39-m08-model-smoke.py": request["helper_sha256"],
        "runner.py": request["helper_sha256"],  # helper snapshot, not executed 41 entry
        "40-m08-kv-backends.py": request["adapter_sha256"],
        "41-m08-model-check.py": request["runner_sha256"],
    }
    for name, expected in source_extra.items():
        assert sha(run / name) == expected, name
    adapter_text = (run / "40-m08-kv-backends.py").read_text(encoding="utf-8")
    runner_text = (run / "41-m08-model-check.py").read_text(encoding="utf-8")
    assert "compiled = kernel[(b,)]" in adapter_text
    assert "FORBIDDEN_ORIGINAL_FALLBACK_IN_TRITON_ARM" in runner_text
    assert "original_calls += 1" in runner_text
    assert "selected.compiled_kernel is None" in runner_text
    report = {"status": "AUDIT_IN_PROGRESS", "gpu_run": False,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "audit_script_sha256": sha(Path(__file__)), "python": sys.version,
              "run": run.name, "source_manifest_entries_verified": len(source),
              "extra_source_hashes_verified": source_extra,
              "request_sha256": sha(run / "request.json"),
              "result_sha256": sha(run / "result.json"),
              "scope": "Independent CPU reload/hash/byte-comparison; 41 eager diagnostic only",
              "arms": {}, "pairs": []}
    arm_arrays, arm_results, arm_outputs = {}, {}, {}
    for arm in ("original_repeat", "original", "triton"):
        root = run / arm
        arm_result = read(root / "result.json")
        assert arm_result["status"] == "MODEL_ARM_CHECKED"
        assert arm_result["input_token_ids"] == request["input_token_ids"]
        assert arm_result["raw_generated_token_ids"] == result["raw_expected_token_ids"]
        assert arm_result["cache_initial_value"] == -7
        assert arm_result["cache_allocation"]["shape"] == [2, 28, 513, 1, 8, 128]
        for name, metadata in arm_result["artifacts"].items():
            path = root / name
            assert path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"], (arm, name)
        artifact_names = {p.name for p in root.iterdir() if p.suffix in (".npz", ".jsonl")}
        assert artifact_names == set(arm_result["artifacts"])
        raw = read(root / "raw-output-tokens.json")
        output = read(root / "outputs.json")
        tokens = [x["token_id"] for x in raw]
        assert len(raw) == 32 and tokens == result["raw_expected_token_ids"]
        assert all(x["uid"] == 0 and x["finished"] == (i == 31) for i, x in enumerate(raw))
        assert len(output) == 1 and output[0]["token_ids"] == tokens
        trace = [json.loads(line) for line in (root / "store-checks.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(trace) == 1792
        counts, hashes = Counter(), Counter()
        visited = {layer: set() for layer in range(28)}
        successful = []
        for call in range(896):
            before, after = trace[call * 2:call * 2 + 2]
            assert before["status"] == "BEFORE_BACKEND" and after["status"] == "FULL_BACKING_AND_READONLY_CHECKED"
            assert before["call_id"] == after["call_id"] == call
            for key in before:
                if key != "status":
                    assert before[key] == after[key], (arm, call, key)
            step, layer = divmod(call, 28)
            phase, batch = ("prefill", 128) if step == 0 else ("decode", 1)
            ids = list(range(128)) if step == 0 else [127 + step]
            inputs = request["input_token_ids"] if step == 0 else [tokens[step - 1]]
            assert after["layer"] == layer and after["phase"] == phase
            assert after["tokens_B"] == batch and after["D"] == 1024
            assert after["indices"] == after["positions"] == ids and after["input_token_ids"] == inputs
            assert len(set(ids)) == batch and all(0 <= i < 513 for i in ids)
            assert after["readonly_allocations"] == 2
            t = after["tensors"]
            for name in ("k_cache", "v_cache", "k", "v", "indices"):
                m = t[name]
                assert m["device"] == "cuda:0"
                assert m["data_ptr"] == m["storage_ptr"] + m["element_size"] * m["storage_offset"]
                assert m["dtype"] == ("torch.int32" if name == "indices" else "torch.float16")
            assert t["k"]["shape"] == t["v"]["shape"] == [batch, 1024]
            assert t["k"]["stride"] == t["v"]["stride"] == [4096, 1]
            assert t["k_cache"]["shape"] == t["v_cache"]["shape"] == [513, 8, 128]
            assert t["k_cache"]["stride"] == t["v_cache"]["stride"] == [1024, 128, 1]
            assert t["indices"]["shape"] == [batch] and t["indices"]["stride"] == [1]
            assert t["k_cache"]["storage_ptr"] == t["v_cache"]["storage_ptr"]
            assert t["k"]["storage_ptr"] == t["v"]["storage_ptr"] != t["k_cache"]["storage_ptr"]
            assert t["indices"]["storage_ptr"] not in (t["k_cache"]["storage_ptr"], t["k"]["storage_ptr"])
            assert t["k_cache"]["storage_offset"] == layer * 513 * 1024
            assert t["v_cache"]["storage_offset"] == (28 + layer) * 513 * 1024
            launch = after["actual_launch"]
            expected = {"backend": "triton" if arm == "triton" else "original", "B": batch, "C": 513, "D": 1024,
                        "input_k_stride": 4096, "input_v_stride": 4096, "cache_k_stride": 1024, "cache_v_stride": 1024,
                        "grid": [batch] if arm == "triton" else None,
                        "BLOCK_D": 1024 if arm == "triton" else None, "num_warps": 4 if arm == "triton" else None}
            assert all(launch[key] == value for key, value in expected.items())
            if arm == "triton":
                assert after["compiled_kernel_type"] == "CompiledKernel"
                assert re.fullmatch(r"[0-9a-f]{64}", after["compiled_kernel_hash"])
                hashes[after["compiled_kernel_hash"]] += 1
            visited[layer].update(ids)
            counts[phase] += 1
            successful.append(after)
        assert counts == {"prefill": 28, "decode": 868}
        assert arm_result["store_calls"] == arm_result["full_cache_checks"] == arm_result["selected_launch_count"] == 896
        assert arm_result["readonly_allocation_checks"] == 1792
        assert arm_result["original_calls"] == (0 if arm == "triton" else 896)
        data = {name: arrays(root / name) for name in sorted(artifact_names) if name.endswith(".npz")}
        sample_checks = []
        for phase in ("prefill", "decode"):
            sample = data[f"layer0-{phase}.npz"]
            ids = sample["indices"]
            expected_ids = list(range(128)) if phase == "prefill" else [128]
            assert ids.dtype == np.int32 and ids.tolist() == expected_ids
            for kind in ("k", "v"):
                reconstructed = sample[f"before_{kind}"].copy().reshape(513, 1024)
                source_kv = sample[kind]
                assert source_kv.dtype == np.float16 and source_kv.shape == (len(ids), 1024)
                for source_row, destination in enumerate(ids):
                    reconstructed[int(destination), :] = source_kv[source_row, :]
                reconstructed = reconstructed.reshape(sample[f"before_{kind}"].shape)
                assert same_bits(reconstructed, sample[f"expected_{kind}"])
                assert same_bits(reconstructed, sample[f"after_{kind}"])
                if phase == "prefill":
                    assert np.all(sample[f"before_{kind}"] == np.float16(-7))
                else:
                    assert same_bits(sample[f"before_{kind}"], data["layer0-prefill.npz"][f"after_{kind}"])
            sample_checks.append({"phase": phase, "rows": len(ids), "full_layer_expected_bytes": 2 * sample["after_k"].nbytes})
        final = data["final-cache.npz"]["cache"]
        assert final.dtype == np.float16 and final.shape == (2, 28, 513, 1, 8, 128)
        assert np.isfinite(final).all()
        untouched_elements = 0
        for layer, locations in visited.items():
            assert locations == set(range(159))
            untouched = sorted(set(range(513)) - locations)
            assert np.all(final[:, layer, untouched] == np.float16(-7))
            untouched_elements += final[:, layer, untouched].size
        for kv_index, kind in enumerate(("k", "v")):
            assert same_bits(final[kv_index, 0, :129], data["layer0-decode.npz"][f"after_{kind}"][:129])
        logits = data["sampled-logits.npz"]
        logit_checks = {}
        for key, token_index in (("first_prefill", 0), ("first_decode", 1), ("last_decode", 31)):
            x = logits[key]
            assert x.shape == (1, 151936) and x.dtype == np.float16 and np.isfinite(x).all()
            best = int(np.argmax(x[0]))
            assert best == tokens[token_index]
            top = np.partition(x[0].astype(np.float32), -2)[-2:]
            logit_checks[key] = {"argmax": best, "top1_top2_gap": float(top.max() - top.min())}
        report["arms"][arm] = {"artifact_hashes_verified": arm_result["artifacts"],
                              "auxiliary_json_sha256": {name: sha(root / name) for name in ("result.json", "outputs.json", "raw-output-tokens.json", "initialization.json")},
                              "jsonl_lines": len(trace), "successful_calls": len(successful), "phase_counts": dict(counts),
                              "original_calls": arm_result["original_calls"], "compiled_kernel_hash_counts": dict(hashes),
                              "saved_samples_independently_reconstructed": sample_checks,
                              "final_untouched_sentinel_elements_checked": untouched_elements,
                              "saved_logits_checks": logit_checks, "raw_steps": 32, "returned_tokens": len(output[0]["token_ids"])}
        arm_arrays[arm], arm_results[arm], arm_outputs[arm] = data, arm_result, output
    total_arrays, total_bytes, total_elements = 0, 0, 0
    for left, right in (("original_repeat", "original"), ("original", "triton")):
        assert arm_outputs[left] == arm_outputs[right]
        assert arm_results[left]["raw_generated_token_ids"] == arm_results[right]["raw_generated_token_ids"]
        pair = {"left": left, "right": right, "returned_output_and_raw_tokens_equal": True, "arrays": []}
        assert set(arm_arrays[left]) == set(arm_arrays[right])
        for filename, left_data in arm_arrays[left].items():
            right_data = arm_arrays[right][filename]
            assert set(left_data) == set(right_data)
            for name, x in left_data.items():
                y = right_data[name]
                assert same_bits(x, y), (left, right, filename, name)
                pair["arrays"].append({"file": filename, "name": name, "shape": list(x.shape), "dtype": str(x.dtype), "bytes_per_side": x.nbytes})
                total_arrays += 1
                total_bytes += x.nbytes
                total_elements += x.size
        report["pairs"].append(pair)
    report.update(status="INDEPENDENT_CPU_EVIDENCE_AUDIT_PASS", total_successful_runtime_records=2688,
                  paired_arrays_compared=total_arrays, paired_bytes_compared_per_side=total_bytes,
                  paired_elements_compared_per_side=total_elements,
                  source_hash_entries_verified=len(source) + len(source_extra),
                  artifact_hash_entries_verified=sum(len(x["artifact_hashes_verified"]) for x in report["arms"].values()),
                  limitations=["Eager-only evidence; no CUDA Graph capture/replay is covered by run 41.",
                               "All 2688 runtime success records and launch metadata were audited. Only the six saved layer0 samples can be independently reconstructed from archived per-call K/V values.",
                               "Final complete cache and three saved logits tensors per arm are byte-compared; intermediate all-layer K/V and all-step logits were not archived.",
                               "Readonly backing checks are source-and-runtime-log evidence, not an offline rerun of each backing allocation.",
                               "CompiledKernel type/hash and real launch counters are retained; this run does not archive PTX/cubin or a kernel profiler trace.",
                               "Checkpoint SHA is recorded by the GPU run; this host CPU audit does not reopen/re-hash the WSL checkpoint.",
                               "No GPU rerun, performance inference, model-quality assessment or independent learner mastery claim."])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    destination = args.run / "evidence-audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing audit; use a new explicitly named result for reruns")
    try:
        report = audit(args.run)
    except BaseException:
        report = {"status": "INDEPENDENT_CPU_EVIDENCE_AUDIT_FAILED", "gpu_run": False, "error": traceback.format_exc()}
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "source_hash_entries_verified", "artifact_hash_entries_verified", "total_successful_runtime_records", "paired_arrays_compared", "paired_bytes_compared_per_side", "paired_elements_compared_per_side")}, indent=2))
