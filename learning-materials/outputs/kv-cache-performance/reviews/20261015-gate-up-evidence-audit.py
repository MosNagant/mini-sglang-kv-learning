"""Independent CPU audit of saved experiment 49 and its experiment 48 fixtures.

Does not import Torch, Triton or experiment modules and never launches a GPU.
Recomputes the FP64 oracles, all saved numerical metrics, timing statistics,
candidate selection and held-out ratios. Runtime-only checks are labelled as
reported evidence rather than represented as saved-array verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

import numpy as np


NAMES = (
    "gemv_direct_r4_w4", "gemv_direct_r8_w4", "gemv_direct_r8_w8",
    "gemv_splitk4_r4_w4", "swiglu_fused_r4_w4", "swiglu_fused_r8_w4",
    "swiglu_packed_r4_w4",
)
STEPS = (1, 16, 31)


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=3e-11, abs_tol=1e-13),
            f"Numeric field mismatch {label}: {actual} != {expected}")


def safe_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Manifest escapes archive: " + str(relative))
    return path


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def metrics(actual, reference, oracle):
    a, r, o = [np.asarray(value, dtype=np.float64) for value in (actual, reference, oracle)]
    if a.shape != r.shape or a.shape != o.shape or any(not np.isfinite(value).all() for value in (a, r, o)):
        return {"pass": False, "reason": "shape or finite check failed"}
    error = np.abs(a - r)
    count = int(np.count_nonzero(error > 0.002 + 0.002 * np.abs(r)))
    norm = max(math.sqrt(float(np.dot(o.ravel(), o.ravel()))), 1e-12)
    da, dr = (a - o).ravel(), (r - o).ravel()
    candidate = math.sqrt(float(np.dot(da, da))) / norm
    original = math.sqrt(float(np.dot(dr, dr))) / norm
    limit = 1.10 * original + 1e-6
    return {"pass": count == 0 and candidate <= limit,
            "violations": count, "max_abs_vs_original": float(error.max()),
            "rmse_vs_original": math.sqrt(float(np.dot(error.ravel(), error.ravel())) / error.size),
            "candidate_nrmse": candidate, "original_nrmse": original, "nrmse_limit": limit,
            "worst_original_index": int(error.argmax()), "element_count": int(a.size)}


def compare_metrics(recorded, expected, label):
    require(set(recorded) == set(expected), "Metric keys differ: " + label)
    for key in expected:
        if isinstance(expected[key], (bool, int, str)):
            require(recorded[key] == expected[key], "Metric differs: " + label + "/" + key)
        else:
            close(recorded[key], expected[key], label + "/" + key)


def audit(run, fixture_run):
    plan, result = read(run / "plan.json"), read(run / "result.json")
    require(result["status"] == "GATE_UP_CAMPAIGN_COMPLETE", "Campaign did not complete")
    require(result["plan_sha256"] == digest(run / "plan.json"), "Plan hash differs")
    require(result["candidate_source_sha256"] == plan["candidate_source_sha256"] == digest(run / "candidates.py"),
            "Candidate source hash differs")
    require([entry["name"] for entry in plan["candidates"]] == list(NAMES), "Frozen seven candidates differ")
    require([entry["kind"] for entry in plan["candidates"]] == ["gemv"] * 4 + ["swiglu"] * 3, "Candidate semantics differ")
    require(plan["timing"] == {"fresh_process_rounds": 3, "interval_samples": 30, "hot_replays": 100, "rotate28_replays": 10},
            "Predeclared timing policy differs")
    require(plan["primary"] == "rotate28_graph_step1_full_GateUp_plus_SwiGLU", "Primary objective differs")
    for key, value in {"all_elements_atol": .002, "rtol": .002, "nrmse_multiplier": 1.10, "nrmse_floor": 1e-6}.items():
        require(plan["numerical_rule"][key] == value, "Numerical rule differs: " + key)
    for relative, expected in plan["frozen_files"].items():
        require(digest(safe_path(run, relative)) == expected, "Frozen file differs: " + relative)
    manifest = read(run / "artifact-manifest.json")
    for relative, expected in manifest.items():
        require(digest(safe_path(run, relative)) == expected, "Raw artifact hash differs: " + relative)
    require({"runner.py", "candidates.py", "plan.json", "result.json", "numerical.json"} <= set(manifest),
            "Artifact manifest omitted primary records")

    fixture, fixture_result, fixture_plan = [read(fixture_run / name) for name in ("fixture.json", "result.json", "plan.json")]
    require(fixture["status"] == fixture_result["status"] == "GATE_UP_FIXTURE_CAPTURE_PASS", "Fixture incomplete")
    require(digest(fixture_run / "fixture.json") == plan["fixture_sha256"] == result["fixture_sha256"] == fixture_result["fixture_sha256"],
            "Fixture manifest hash differs")
    require(digest(fixture_run / "plan.json") == fixture_result["plan_sha256"] == fixture["plan_sha256"], "Fixture plan differs")
    require(fixture["capture_steps_zero_based"] == list(STEPS) and fixture["layer_count"] == 28, "Fixture scope differs")
    require(fixture["weight_count"] == len(fixture["weights"]) == 28 and fixture["sample_count"] == len(fixture["samples"]) == 84,
            "Fixture counts differ")
    require(fixture["raw_generated_token_ids"] == fixture["expected_raw_token_ids"] == fixture_plan["raw_expected_token_ids"],
            "Fixture token output differs from accepted input contract")
    require(len(fixture["raw_generated_token_ids"]) == 32 and len(fixture["input_token_ids"]) == 128, "Workload size differs")
    require(fixture["observed_gate_up_calls"] == fixture["observed_activation_calls"] == 896, "Runtime call counts differ")
    for relative, expected in fixture_plan["frozen_files"].items():
        require(digest(safe_path(fixture_run, relative)) == expected, "Fixture frozen source differs: " + relative)
    for relative, expected in fixture_plan["source_manifest"].items():
        require(digest(safe_path(fixture_run / "source", relative)) == expected, "Original model source differs: " + relative)
    sample_map = {}
    for item in fixture["weights"] + fixture["samples"]:
        path = safe_path(fixture_run, item["path"])
        require(digest(path) == item["sha256"] and path.stat().st_size == item["file_bytes"], "Fixture payload hash/size differs")
        if "step" in item:
            key = item["step"], item["layer"]
            require(key not in sample_map, "Duplicate fixture sample")
            require(item["actual_input_token_ids"] == [fixture["raw_generated_token_ids"][item["step"] - 1]], "Decode token alignment differs")
            value = arrays(path)
            require(set(value) == {"x", "z", "y"}, "Captured array keys differ")
            for name, shape in {"x": (1, 1024), "z": (1, 6144), "y": (1, 3072)}.items():
                require(value[name].shape == shape and value[name].dtype == np.float16 and np.isfinite(value[name]).all(), "Captured shape/dtype/finite differs")
            sample_map[key] = value
    require(set(sample_map) == {(step, layer) for step in STEPS for layer in range(28)}, "Fixture matrix incomplete")

    numerical = read(run / "numerical.json")
    records = numerical["records"]
    require(numerical["status"] == "NUMERICAL_CAMPAIGN_COMPLETE" and numerical["case_count"] == len(records) == 588, "Numerical case count differs")
    require(lines(run / "numerical/cases.jsonl") == records, "Incremental numerical records differ")
    keyed = {(r["name"], r["step"], r["layer"]): r for r in records}
    require(len(keyed) == 588 and set(keyed) == {(name, step, layer) for name in NAMES for step in STEPS for layer in range(28)}, "Numerical matrix incomplete")
    checks, eligible = {}, {name: True for name in NAMES}
    numerical_arrays = oracle_arrays = numerical_elements = 0
    oracle_max_abs_difference = {"z64": 0., "y64": 0.}
    for weight_entry in sorted(fixture["weights"], key=lambda item: item["layer"]):
        layer = weight_entry["layer"]
        weight = np.load(fixture_run / weight_entry["path"], allow_pickle=False)
        require(weight.shape == (6144, 1024) and weight.dtype == np.float16 and np.isfinite(weight).all(), "Weight shape/dtype/finite differs")
        w64 = weight.astype(np.float64)
        for step in STEPS:
            sample = sample_map[step, layer]
            # Use explicit elementwise products/reduction rather than the runner's
            # BLAS matmul. FP16 products are promoted before arithmetic.
            dot = np.sum(w64 * sample["x"].astype(np.float64), axis=1, dtype=np.float64)[None, :]
            intermediate = dot.astype(np.float16).astype(np.float64)
            gate, up = intermediate[:, :3072], intermediate[:, 3072:]
            # logaddexp is a separate stable expression for the sigmoid.
            activated = gate * np.exp(-np.logaddexp(0., -gate)) * up
            stored = arrays(run / "numerical" / f"oracle-step{step:02d}-layer{layer:02d}.npz")
            require(set(stored) == {"z64", "y64"}, "Oracle array keys differ")
            for label, expected in (("z64", dot), ("y64", activated)):
                require(stored[label].dtype == np.float64 and stored[label].shape == expected.shape and np.isfinite(stored[label]).all(), "Oracle shape/dtype/finite differs")
                difference = float(np.max(np.abs(stored[label] - expected)))
                oracle_max_abs_difference[label] = max(oracle_max_abs_difference[label], difference)
                require(np.allclose(stored[label], expected, rtol=2e-12, atol=2e-12), "Independent FP64 oracle mismatch: " + label)
                oracle_arrays += 1
            for name in NAMES:
                record = keyed[name, step, layer]
                payload = arrays(run / "numerical" / name / f"step{step:02d}-layer{layer:02d}.npz")
                expected_keys = {"y", "z"} if name.startswith("gemv_") else {"y"}
                require(set(payload) == expected_keys and set(record["checks"]) == expected_keys, "Candidate saved output keys differ")
                current = {}
                for output in expected_keys:
                    actual = payload[output]
                    require(actual.dtype == np.float16, "Candidate output dtype differs")
                    current[output] = metrics(actual, sample[output], stored[output + "64"])
                    compare_metrics(record["checks"][output], current[output], f"{name}/{step}/{layer}/{output}")
                    numerical_arrays += 1
                    numerical_elements += actual.size
                computed_pass = all(value["pass"] for value in current.values()) and record["readonly"] and record["packing_roundtrip"]
                require(record["pass"] == computed_pass, "Case eligibility differs")
                require("error" not in record, "Saved candidate exception requires investigation")
                eligible[name] &= computed_pass
                checks[name, step, layer] = current["y"]
            checks["original", step, layer] = metrics(sample["y"], sample["y"], stored["y64"])
        print(f"AUDIT_NUMERICAL_LAYER={layer + 1}/28", flush=True)
    eligible_names = [name for name in NAMES if eligible[name]]
    require(numerical["eligible"] == eligible_names and numerical["failed"] == [name for name in NAMES if not eligible[name]], "Numerical eligible/failed lists differ")
    require(result["numerical_eligible"] == eligible_names, "Campaign eligible list differs")

    interval_count = 0
    postcheck_count = 0
    timing_results = {}
    for task_path in sorted(run.glob("*/*/task.json")):
        folder, task = task_path.parent, read(task_path)
        record = read(folder / "result.json")
        require(record["status"] == "TIMING_WORKER_PASS" and record["task"] == task, "Timing worker did not pass")
        require(record["candidate_source_sha256"] == plan["candidate_source_sha256"], "Timing candidate source differs")
        require(record["readonly"] and record["graph_outputs_checked"] and record["graph_nan_reset_and_changed_input_controls"], "Runtime correctness flags incomplete")
        interval_records = lines(folder / "intervals.jsonl")
        require(interval_records == record["samples"], "Raw incremental intervals differ from result")
        arms = task["arms"]
        expected_order = []
        for setting in ("hot_l0", "rotate28"):
            for policy in ("graph", "eager"):
                for sample in range(30):
                    rotation = (sample + task["round"]) % len(arms)
                    order = arms[rotation:] + arms[:rotation]
                    if sample % 2:
                        order = order[::-1]
                    expected_order.extend((arm, setting, policy, sample) for arm in order)
        require([(r["arm"], r["setting"], r["policy"], r["sample"]) for r in interval_records] == expected_order, "Interval matrix/order differs")
        bins = {}
        for item in interval_records:
            repeats, calls = (100, 1) if item["setting"] == "hot_l0" else (10, 28)
            require(item["replays"] == repeats and item["calls_per_replay"] == calls, "Timing divisor differs")
            require(math.isfinite(item["interval_ms"]) and item["interval_ms"] > 0., "Nonpositive/nonfinite raw interval")
            value = item["interval_ms"] * 1000. / (repeats * calls)
            close(item["us_per_pipeline"], value, "interval conversion")
            bins.setdefault((item["arm"], item["setting"], item["policy"]), []).append(value)
        for (arm, setting, policy), values in bins.items():
            require(len(values) == 30, "Missing timing samples")
            close(record["medians"][arm][setting][policy], statistics.median(values), "median")
        final = lines(folder / "post-timing-checks.jsonl")
        require(final == record["post_timing_checks"], "Post-timing incremental checks differ")
        require(Counter((r["arm"], r["policy"], r["layer"]) for r in final) == Counter((arm, policy, layer) for arm in arms for policy in ("eager", "graph") for layer in range(28)), "Post-timing check matrix incomplete")
        for item in final:
            # These final GPU values were not archived. Confirm the runtime
            # metrics agree with independently audited deterministic fixtures;
            # do not call this direct verification of final GPU arrays.
            compare_metrics(item["check"], checks[item["arm"], task["step"], item["layer"]], "reported post-timing metrics")
            require(item["check"]["pass"], "Post-timing numeric rejection")
        interval_count += len(interval_records)
        postcheck_count += len(final)
        key = task["stage"], task["step"], task["round"]
        require(key not in timing_results, "Duplicate timing worker")
        timing_results[key] = record

    expected_tasks = {("selection", 1, round_id) for round_id in (1, 2, 3)} | {("holdout", step, round_id) for step in (16, 31) for round_id in (1, 2, 3)}
    require(eligible_names and set(timing_results) == expected_tasks, "Completed campaign timing task set differs")
    ratios = {}
    for name in eligible_names:
        ratios[name] = []
        for round_id in (1, 2, 3):
            worker = timing_results["selection", 1, round_id]
            require(worker["task"]["arms"] == ["original"] + eligible_names, "Selection timed ineligible/missing candidate")
            ratios[name].append(worker["medians"]["original"]["rotate28"]["graph"] / worker["medians"][name]["rotate28"]["graph"])
    selected = max(eligible_names, key=lambda name: statistics.median(ratios[name]))
    selection = read(run / "selection.json")
    require(selection == result["selection"] and digest(run / "selection.json") == result["selection_sha256"], "Frozen selection hash/content differs")
    require(selected == selection["selected_candidate"] == result["selected_candidate"], "Selected winner differs")
    require(set(selection["ratios_all"]) == set(ratios), "Selection ratio keys differ")
    for name, values in ratios.items():
        for actual, expected in zip(selection["ratios_all"][name], values):
            close(actual, expected, "paired process ratio")
    require(selection["per_round_speedups"] == selection["ratios_all"][selected], "Selected per-round ratios differ")
    close(selection["representative_speedup"], statistics.median(ratios[selected]), "selection representative")
    reductions = [1. - 1. / value for value in ratios[selected]]
    strong = statistics.median(reductions) >= .10 and min(reductions) >= .05
    require(selection["strong_local_gain"] == strong, "Predeclared strong-gain criterion differs")
    holdout = read(run / "holdout.json")
    require(holdout == result["holdout"] and holdout["status"] == "PASS", "Holdout collection incomplete")
    holdout_ratios = {}
    for step in (16, 31):
        values = []
        for round_id in (1, 2, 3):
            worker = timing_results["holdout", step, round_id]
            require(worker["task"]["arms"] == ["original", selected], "Holdout changed selected arm")
            values.append(worker["medians"]["original"]["rotate28"]["graph"] / worker["medians"][selected]["rotate28"]["graph"])
        require(len(holdout["per_step_ratios"][str(step)]) == 3, "Holdout ratio count differs")
        for actual, expected in zip(holdout["per_step_ratios"][str(step)], values):
            close(actual, expected, "holdout ratio")
        holdout_ratios[str(step)] = values

    compiled = []
    for path in sorted(run.rglob("compiled.json")):
        entries = read(path)
        for entry in entries:
            for relative, expected in entry["files"].items():
                require(digest(safe_path(path.parent, relative)) == expected, "Compiled artifact hash differs")
            compiled.append({"record": path.relative_to(run).as_posix(), "kernel_index": entry["kernel_index"],
                             "name": entry["name"], "registers_per_thread": entry["n_regs"],
                             "shared_memory_bytes": entry["shared_bytes"], "n_spills_raw": entry["n_spills_raw"],
                             "artifact_kinds": [Path(name).suffix[1:] for name in entry["files"]]})
    changed_controls = []
    for step in STEPS:
        alternate = 16 if step == 1 else 1
        current, other = sample_map[step, 0], sample_map[alternate, 0]
        require(np.any(current["x"] != other["x"]), "Changed-input control had identical values")
        differences = np.abs(current["y"].astype(np.float64) - other["y"].astype(np.float64))
        bad = int(np.count_nonzero(differences > .002 + .002 * np.abs(other["y"].astype(np.float64))))
        require(bad > 0, "Stale output can impersonate alternate input control")
        changed_controls.append({"step": step, "alternate": alternate, "stale_y_rejected_elements": bad})

    performance_table = []
    setup_table = []
    for stage, step in (("selection", 1), ("holdout", 16), ("holdout", 31)):
        workers = [timing_results[stage, step, round_id] for round_id in (1, 2, 3)]
        arms = workers[0]["task"]["arms"]
        for arm in arms:
            for setting in ("hot_l0", "rotate28"):
                for policy in ("graph", "eager"):
                    baseline = [worker["medians"]["original"][setting][policy] for worker in workers]
                    current = [worker["medians"][arm][setting][policy] for worker in workers]
                    paired = [base / candidate for base, candidate in zip(baseline, current)]
                    declines = [100. * (1. - candidate / base) for base, candidate in zip(baseline, current)]
                    performance_table.append({"stage": stage, "step": step, "arm": arm, "setting": setting,
                        "policy": policy, "original_process_medians_us": baseline,
                        "arm_process_medians_us": current, "paired_original_over_arm": paired,
                        "paired_latency_reduction_percent": declines,
                        "median_paired_speedup": statistics.median(paired),
                        "median_paired_latency_reduction_percent": statistics.median(declines),
                        "rounds_faster_than_original": sum(value > 1. for value in paired),
                        "is_selection_primary": stage == "selection" and setting == "rotate28" and policy == "graph",
                        "limit": "hot_l0 is diagnostic and cannot replace the predeclared rotate28 primary; Eager allocation policies differ"})
            setup_table.append({"stage": stage, "step": step, "arm": arm,
                                "per_round_setup": [worker["setup"][arm] for worker in workers]})
    numerical_summary = {}
    for name in NAMES:
        current = [record for record in records if record["name"] == name]
        numerical_summary[name] = {
            "cases": len(current), "passed": sum(record["pass"] for record in current),
            "output_metrics": {output: {
                "max_abs_vs_original": max(record["checks"][output]["max_abs_vs_original"] for record in current),
                "max_candidate_nrmse": max(record["checks"][output]["candidate_nrmse"] for record in current),
                "max_original_nrmse": max(record["checks"][output]["original_nrmse"] for record in current),
                "violations_total": sum(record["checks"][output]["violations"] for record in current),
            } for output in current[0]["checks"]},
        }

    return {"status": "INDEPENDENT_GATE_UP_EVIDENCE_AUDIT_PASS", "run": str(run), "fixture_run": str(fixture_run),
            "created_utc": datetime.now(timezone.utc).isoformat(), "auditor_sha256": digest(__file__),
            "sources": {"plan": digest(run / "plan.json"), "runner": digest(run / "runner.py"),
                        "candidates": digest(run / "candidates.py"), "artifact_manifest": digest(run / "artifact-manifest.json"),
                        "fixture_manifest": digest(fixture_run / "fixture.json")},
            "verified_artifact_hashes": len(manifest), "fixture_weights": 28, "fixture_samples": 84,
            "numerical_cases": 588, "numerical_saved_arrays": numerical_arrays, "numerical_saved_elements": numerical_elements,
            "independent_fp64_oracle_arrays": oracle_arrays, "oracle_max_abs_difference": oracle_max_abs_difference,
            "eligible": eligible_names, "fresh_timing_process_results": len(timing_results), "interval_averages": interval_count,
            "post_timing_reported_checks": postcheck_count, "selected_candidate": selected,
            "selection_ratios_all": ratios, "selection_latency_reductions": reductions,
            "strong_local_gain": strong, "holdout_ratios": holdout_ratios,
            "all_settings_performance_table": performance_table, "setup_memory_and_time_table": setup_table,
            "numerical_summary": numerical_summary,
            "changed_input_control_non_degeneracy": changed_controls, "compiled_static_resources": compiled,
            "limitations": [
                "CPU reanalysis only; no GPU execution, new speedup measurements or hardware-counter evidence.",
                "Saved 588 candidate output files and 84 original fixture/oracle pairs were directly rechecked elementwise.",
                "Read-only GPU comparisons, packing roundtrip, Graph sentinel/replay controls and post-timing outputs are runtime-reported; no saved final GPU arrays exist. Their recorded metrics/counts agree with audited fixture metrics.",
                "Selection/holdout contents and hashes are checked; chronological freeze-before-holdout is supported by frozen controller source, not an externally signed timestamp.",
                "NRMSE and tolerances are the frozen scoped engineering contract, not evidence of general model quality.",
                "Timing is the full GateUp plus activation pipeline. Eager allocating baseline versus reused candidate outputs remains an implementation-policy difference.",
                "Static compiler resources are not achieved occupancy, bandwidth or a demonstrated performance cause.",
                "Holdout PASS means collection completed; performance acceptance and whole-model evidence are separate.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fixture-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if args.fixture_run:
        fixture = args.fixture_run.resolve(strict=True)
    else:
        original = Path(read(run / "plan.json")["fixture_run"])
        fixture = original if original.exists() else run.parent.parent / "m08-gate-up-fixture-results" / original.name
        fixture = fixture.resolve(strict=True)
    output = args.output or run / "summaryaudit.json"
    try:
        result = audit(run, fixture)
    except BaseException:
        result = {"status": "INDEPENDENT_GATE_UP_EVIDENCE_AUDIT_FAILED", "run": str(run),
                  "fixture_run": str(fixture), "auditor_sha256": digest(__file__), "error": traceback.format_exc()}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "numerical_cases", "eligible", "interval_averages", "selected_candidate", "strong_local_gain", "holdout_ratios")}, ensure_ascii=False))
    print("SAVED=" + str(output))


if __name__ == "__main__":
    main()
