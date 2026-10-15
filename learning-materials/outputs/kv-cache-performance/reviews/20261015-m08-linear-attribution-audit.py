"""Independent offline SQL/data audit of one saved 47 Eager attribution run.

This does not import or call the production analyzer. It uses direct SQL joins
and reconstructs layer/shape/phase expectations independently from model config.
No GPU, Torch, model load, or WSL command is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def verify_hashes(root, manifest):
    root = Path(root).resolve()
    for name, digest in manifest.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "manifest escapes root: " + name)
        require(path.is_file() and sha(path) == digest, "hash mismatch: " + str(path))
    return len(manifest)


def audit(run, accepted_timing=None):
    run = Path(run).resolve()
    plan, result, workload = (read(run / name) for name in ("plan.json", "result.json", "workload.json"))
    report = read(run / "attribution.json")
    require(result["status"] == "LINEAR_ATTRIBUTION_COMPLETE", "47 did not complete")
    require(workload["status"] == "LINEAR_ATTRIBUTION_WORKLOAD_CHECKED", "workload not checked")
    require(report["status"] == "LINEAR_KERNEL_ATTRIBUTION_COMPLETE", "attribution not complete")
    require(result["attribution"] == report, "result/attribution mismatch")
    require(all(x["policy"] == "eager" for x in (result, workload, report)), "policy is not Eager")
    require(result["backend"] == workload["backend"] == "original", "backend differs")
    require(result["not_timing_sample"] is True and workload["not_timing_sample"] is True,
            "instrumented run mislabelled as timing")
    require(result["profile_returncode"] == 0, "profile command failed")
    require(result["plan_sha256"] == sha(run / "plan.json"), "plan hash differs")
    require(plan["attribution_policy"] == "eager" and workload["engine_options"]["cuda_graph_bs"] == [],
            "Graph enabled")
    source_manifest = read(run / "source-manifest.json")
    require(source_manifest == plan["source_manifest"], "source manifests disagree")
    sources_checked = verify_hashes(run / "source", source_manifest)
    frozen_checked = verify_hashes(run, plan["frozen_files"])
    artifacts_checked = verify_hashes(run, result["artifact_sha256"])
    require(report["analyzer_sha256"] == sha(run / "47-analyze-linear-attribution.py"),
            "saved analyzer hash differs")
    verify_hashes(run, report["source_hashes"])

    accepted_plan, accepted_summary = read(run / "accepted-timing-plan.json"), read(run / "accepted-timing-summary.json")
    require(sha(run / "accepted-timing-plan.json") == plan["accepted_timing_plan_sha256"], "accepted plan copy hash")
    require(sha(run / "accepted-timing-summary.json") == plan["accepted_timing_summary_sha256"], "accepted summary copy hash")
    if accepted_timing is None:
        accepted_name = PurePosixPath(plan["accepted_timing"].replace("\\", "/")).name
        accepted_timing = run.parents[1] / "m08-model-timing-results" / accepted_name
    accepted_timing = Path(accepted_timing).resolve()
    require(sha(accepted_timing / "plan.json") == plan["accepted_timing_plan_sha256"], "original42 plan differs")
    require(sha(accepted_timing / "summary.json") == plan["accepted_timing_summary_sha256"], "original42 summary differs")
    require(accepted_summary["status"] == "MODEL_TIMING_COLLECTION_COMPLETE", "accepted42 incomplete")
    require(plan["source_manifest"] == accepted_plan["source_manifest"], "source changed from42")
    require(plan["input_token_ids"] == accepted_plan["input_token_ids"] and len(plan["input_token_ids"]) == 128,
            "input token workload changed")
    expected_tokens = accepted_plan["raw_expected_token_ids"]
    require(plan["max_new_tokens"] == 32 and len(expected_tokens) == 32, "output token count changed")
    require(plan["raw_expected_token_ids"] == workload["raw_generated_token_ids"] == expected_tokens,
            "actual generated tokens differ from accepted42")
    require(len(workload["output"]) == 1 and workload["output"][0]["token_ids"] == expected_tokens,
            "returned output tokens differ")
    require(plan["model"]["revision"] == accepted_plan["model"]["revision"], "model revision changed")
    require([(w["name"], w["sha256"]) for w in plan["model"]["weights"]] ==
            [(w["name"], w["sha256"]) for w in accepted_plan["model"]["weights"]], "model manifest changed")
    require(workload["environment"]["packages"] == plan["timing_packages"], "package versions differ")

    config = plan["model"]["config"]
    layers, hidden, head_dim = config["num_hidden_layers"], config["hidden_size"], config["head_dim"]
    qwidth, kvwidth = config["num_attention_heads"] * head_dim, config["num_key_value_heads"] * head_dim
    inter, vocab = config["intermediate_size"], config["vocab_size"]
    require((layers, hidden, qwidth, kvwidth, inter, vocab) == (28, 1024, 2048, 1024, 3072, 151936),
            "unreviewed model dimensions")
    dimensions = {"qkv_proj": [qwidth + 2 * kvwidth, hidden], "o_proj": [hidden, qwidth],
                  "gate_up_proj": [2 * inter, hidden], "down_proj": [hidden, inter], "lm_head": [vocab, hidden]}
    expected_paths = []
    for layer in range(layers):
        for branch, component in (("self_attn", "qkv_proj"), ("self_attn", "o_proj"),
                                  ("mlp", "gate_up_proj"), ("mlp", "down_proj")):
            expected_paths.append((f"model.layers.{layer}.{branch}.{component}", component, layer))
    expected_paths.append(("lm_head", "lm_head", None))
    registry, calls = read(run / "linear-registry.json"), read(run / "linear-calls.json")
    require(len(registry) == 113 and len(calls) == 113 * 32 == 3616, "wrong registry/call count")
    by_path = {entry["module_path"]: entry for entry in registry}
    require(len(by_path) == len(registry) and set(by_path) == {x[0] for x in expected_paths}, "registry paths differ")
    require(len({entry["weight_object_id"] for entry in registry}) == 113, "duplicate registered weight objects")
    for path, component, layer in expected_paths:
        entry = by_path[path]
        require(entry["component"] == component and entry["layer"] == layer, "registry role/layer mismatch")
        require(entry["weight"]["shape"] == dimensions[component] and entry["weight"]["dtype"] == "torch.float16"
                and entry["weight"]["device"] == "cuda:0" and entry["bias"] is None, "registry weight metadata mismatch")
        if component == "lm_head":
            require(config["tie_word_embeddings"] is True and entry["effective_weight_path"] == "model.embed_tokens.weight",
                    "LM head tie evidence missing")
    phase_counts, role_shape_counts = Counter(), Counter()
    by_label = {}
    checked_source_lines = set()
    for step in range(32):
        phase = "prefill" if step == 0 else "decode"
        for slot, (path, component, layer) in enumerate(expected_paths):
            index = step * 113 + slot
            call = calls[index]
            label = f"M08_LINEAR_{index:06d}"
            rows = 128 if step == 0 and component != "lm_head" else 1
            require(call["call_id"] == index and call["label"] == label and call["status"] == "RETURNED", "call identity/status mismatch")
            require((call["module_path"], call["component"], call["layer"], call["step"], call["phase"]) ==
                    (path, component, layer, step, phase), "call order/phase/role mismatch")
            require(call["weight"] == by_path[path]["weight"] and call["bias"] is None, "weight identity/metadata changed")
            require(call["input"]["shape"] == [rows, dimensions[component][1]] and
                    call["output"]["shape"] == [rows, dimensions[component][0]], "linear dimensions disagree")
            require(all(call[key]["dtype"] == "torch.float16" and call[key]["device"] == "cuda:0"
                        for key in ("input", "weight", "output")), "call dtype/device mismatch")
            relative_source = call["source_file"].split("/source/")[-1]
            source_lines = (run / "source" / relative_source).read_text(encoding="utf-8").splitlines()
            line = source_lines[call["source_line"] - 1]
            require("F.linear" in line, "reported callsite is not an F.linear invocation")
            checked_source_lines.add((relative_source, call["source_line"]))
            by_label[label] = call
            phase_counts[f"{phase}/{component}"] += 1
            role_shape_counts[(phase, component, tuple(call["input"]["shape"]), tuple(call["weight"]["shape"]))] += 1
    require(dict(phase_counts) == workload["observed_phase_component_counts"], "reported phase counts differ")

    db = sqlite3.connect((run / "original-model.sqlite").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    api_tables = [name for name in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER") if name in tables]
    api_union = " UNION ALL ".join(
        f"SELECT '{name}' AS api_source,start,end,globalTid,correlationId,nameId FROM {name}" for name in api_tables)
    label_sql = "SELECT n.rowid AS nvid,n.start,n.end,n.globalTid,n.endGlobalTid,COALESCE(n.text,s.value) AS label FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id"
    labelled_ranges = [dict(r) for r in db.execute(label_sql) if r["label"] in by_label]
    require(len(labelled_ranges) == 3616 and {r["label"] for r in labelled_ranges} == set(by_label), "raw NVTX labels differ")
    require(all(r["end"] is not None and r["end"] > r["start"] and r["endGlobalTid"] in (None, r["globalTid"])
                for r in labelled_ranges), "invalid/cross-thread labelled range")
    # Pairwise interval condition implemented in SQL rather than production bisect.
    overlap_sql = f"WITH n AS ({label_sql}) SELECT COUNT(*) FROM n a JOIN n b ON a.nvid<b.nvid AND a.globalTid=b.globalTid AND a.start<b.end AND b.start<a.end WHERE a.label LIKE 'M08_LINEAR_%' AND b.label LIKE 'M08_LINEAR_%'"
    require(db.execute(overlap_sql).fetchone()[0] == 0, "overlapping labelled ranges")
    duplicate_sql = f"WITH a AS ({api_union}) SELECT COUNT(*) FROM (SELECT (globalTid >> 24) AS process,correlationId,COUNT(*) AS records FROM a GROUP BY process,correlationId HAVING records>1)"
    duplicate_api_keys = db.execute(duplicate_sql).fetchone()[0]
    # This concrete export has unique correlation IDs; reject rather than guess
    # if a future capture needs runtime/driver nesting adjudication.
    require(duplicate_api_keys == 0, "this independent audit requires unique API correlations")
    join_sql = f"""WITH a AS ({api_union}),n AS ({label_sql})
        SELECT k.rowid AS kid,k.start AS gpu_start,k.end AS gpu_end,k.globalPid,k.correlationId,
               k.streamId,k.deviceId,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
               s.value AS kernel_name,a.api_source,a.start AS api_start,a.end AS api_end,
               a.globalTid,a.nameId,n.label,n.start AS nvtx_start,n.end AS nvtx_end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN a ON (a.globalTid >> 24)=(k.globalPid >> 24) AND a.correlationId=k.correlationId
        JOIN n ON n.globalTid=a.globalTid AND n.start<=a.start AND a.end<=n.end
        JOIN StringIds s ON k.demangledName=s.id
        WHERE n.label LIKE 'M08_LINEAR_%'
        ORDER BY k.start,k.end,k.rowid"""
    joined = [dict(row) for row in db.execute(join_sql)]
    require(len(joined) == 3616 and len({r["kid"] for r in joined}) == 3616, "raw SQL join not one kernel per labelled call")
    require(Counter(r["label"] for r in joined) == Counter(by_label.keys()), "raw join labels missing/duplicated")
    kernel_sql = "SELECT k.rowid AS kid,k.*,s.value AS kernel_name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start,k.end,k.rowid"
    kernels = [dict(row) for row in db.execute(kernel_sql)]
    require(len(kernels) == 11945 and all(r["end"] > r["start"] for r in kernels), "kernel count/durations differ")
    require(all(r["graphNodeId"] in (None, 0) for r in kernels), "Graph node in Eager attribution")
    require(all((r["globalPid"] & ((1 << 24) - 1)) == 0 for r in kernels), "unreviewed encoded globalPid")
    joined_ids = {r["kid"] for r in joined}
    unmatched = [r for r in kernels if r["kid"] not in joined_ids]
    unknown_matrix = [r for r in unmatched if any(word in r["kernel_name"].lower() for word in ("gemv", "gemm", "cutlass"))]
    require(not unknown_matrix, "unattributed named matrix kernel")
    all_ns = sum(r["end"] - r["start"] for r in kernels)
    linear_ns = sum(r["gpu_end"] - r["gpu_start"] for r in joined)
    kv = [r for r in kernels if "store_kv_cache<" in r["kernel_name"]]
    require((all_ns, linear_ns, len(kv), sum(r["end"] - r["start"] for r in kv)) ==
            (103501109, 88978321, 896, 870507), "raw totals do not match reviewed run")
    counters = {"linear_calls": len(calls), "linear_nvtx_ranges": len(labelled_ranges),
                "cpu_api_records": sum(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in api_tables),
                "kernel_count": len(kernels), "matched_linear_kernels": len(joined), "unmatched_other_kernels": len(unmatched),
                "all_kernel_total_ns": all_ns, "linear_kernel_total_ns": linear_ns,
                "kv_kernel_count": len(kv), "kv_kernel_total_ns": sum(r["end"] - r["start"] for r in kv)}
    require(all(report[key] == value for key, value in counters.items()), "reported counts/totals disagree")
    require(report["per_call_kernel_count_histogram"] == {"1": 3616}, "kernel/call histogram differs")

    groups = {}
    for key, count in role_shape_counts.items():
        phase, component, input_shape, weight_shape = key
        groups[key] = dict(phase=phase, role=component, input_shape=list(input_shape), weight_shape=list(weight_shape),
                           call_count=count, kernel_count=0, total_ns=0, kernel_names={})
    for row in joined:
        call = by_label[row["label"]]
        key = (call["phase"], call["component"], tuple(call["input"]["shape"]), tuple(call["weight"]["shape"]))
        g = groups[key]
        duration = row["gpu_end"] - row["gpu_start"]
        g["kernel_count"] += 1
        g["total_ns"] += duration
        k = g["kernel_names"].setdefault(row["kernel_name"], {"count": 0, "total_ns": 0})
        k["count"] += 1
        k["total_ns"] += duration
    for g in groups.values():
        g["share_of_all_kernel_duration_sum_pct"] = 100 * g["total_ns"] / all_ns
    reported_groups = {(g["phase"], g["role"], tuple(g["input_shape"]), tuple(g["weight_shape"])): g for g in report["groups"]}
    require(groups == reported_groups, "role/phase/shape aggregate differs from raw SQL")
    kernel_groups = {}
    for row in kernels:
        k = kernel_groups.setdefault(row["kernel_name"], {"count": 0, "total_ns": 0})
        k["count"] += 1
        k["total_ns"] += row["end"] - row["start"]
    require(kernel_groups == report["kernels"], "all-kernel aggregates differ")
    recorded = read_jsonl(run / "attributed-kernels.jsonl")
    recorded_other = read_jsonl(run / "unmatched-kernels.jsonl")
    require(len(recorded) == len(joined) and len(recorded_other) == len(unmatched), "JSONL count mismatch")
    for sqlrow, saved in zip(joined, recorded):
        require((saved["label"], saved["name"], saved["start_ns"], saved["end_ns"], saved["globalPid"], saved["correlationId"]) ==
                (sqlrow["label"], sqlrow["kernel_name"], sqlrow["gpu_start"], sqlrow["gpu_end"], sqlrow["globalPid"], sqlrow["correlationId"]),
                "attributed JSONL identity differs")
        require(saved["call"] == by_label[saved["label"]], "JSONL call metadata differs")
        require(len(saved["api_matches"]) == 1 and saved["api_matches"][0]["globalTid"] == sqlrow["globalTid"],
                "JSONL API thread/correlation multiplicity differs")
    for sqlrow, saved in zip(unmatched, recorded_other):
        require((saved["name"], saved["start_ns"], saved["end_ns"], saved["correlationId"]) ==
                (sqlrow["kernel_name"], sqlrow["start"], sqlrow["end"], sqlrow["correlationId"]), "unmatched JSONL identity differs")
    db.close()
    return {
        "status": "INDEPENDENT_LINEAR_ATTRIBUTION_EVIDENCE_AUDIT_PASS", "run": str(run), "new_gpu_runs": 0,
        "method": "Direct SQL kernel/process/correlation -> API -> same-thread NVTX containment, plus independent metadata sequence and aggregate reconstruction. Production analyzer not imported or executed.",
        "audit_script_sha256": sha(__file__), "source_files_checked": sources_checked,
        "frozen_diagnostic_files_checked": frozen_checked, "result_artifact_hashes_checked": artifacts_checked,
        "source_sha256": {name: sha(run / name) for name in ("original-model.sqlite", "linear-calls.json", "linear-registry.json", "plan.json", "result.json", "attribution.json")},
        "accepted42_direct_source": str(accepted_timing), "accepted42_plan_and_summary_hashes_match": True,
        "raw_tokens_equal_accepted42": True, "raw_tokens_checked": len(expected_tokens),
        "registry_entries_checked": len(registry), "actual_call_records_checked": len(calls),
        "actual_callsite_lines_checked": [list(x) for x in sorted(checked_source_lines)],
        "phase_component_counts": dict(sorted(phase_counts.items())),
        "raw_sql_recomputed_counts": counters, "duplicate_process_correlation_keys": duplicate_api_keys,
        "ambiguous_or_missing_linear_labels": 0, "unattributed_named_matrix_kernels": len(unknown_matrix),
        "graph_kernels": 0,
        "linear_kernels_whose_gpu_execution_is_outside_cpu_nvtx": sum(not (r["nvtx_start"] <= r["gpu_start"] and r["gpu_end"] <= r["nvtx_end"]) for r in joined),
        "recomputed_groups": sorted(groups.values(), key=lambda x: x["total_ns"], reverse=True),
        "independent_join_sql": join_sql, "independent_overlap_sql": overlap_sql,
        "limits": [
            "One instrumented original-backend Eager observation; duration shares do not attribute old43 Graph nodes or establish speedup.",
            "Token equality was independently checked from saved actual IDs; full KV/logits arrays were not captured in this attribution experiment.",
            "Loaded tensor identity/shapes follow saved runtime metadata and frozen instrumentation; model weights were not reloaded by this CPU audit.",
            "Summed GPU kernel durations are not wall-clock critical-path shares or Amdahl bounds.",
            "No occupancy, bank-conflict, cache, bandwidth or AMD hardware-counter evidence was created.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--accepted-timing", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.accepted_timing)
    output = args.output or args.run / "evidence-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "registry": result["registry_entries_checked"], "calls": result["actual_call_records_checked"],
                      "raw_sql_counts": result["raw_sql_recomputed_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
