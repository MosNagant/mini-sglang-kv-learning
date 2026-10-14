"""Read-only SQLite/source audit of 43 profile; no Torch/NSYS/GPU invocation."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def audit(run):
    previous = run.parent / "run-rv7o7fmk"
    previous_before = tree_hashes(previous)
    result, plan, workload = (read(run / name) for name in ("result.json", "plan.json", "workload.json"))
    assert result["status"] == "REAL_MODEL_KV_SHARE_RECORDED" and result["profile_returncode"] == 0
    assert result["backend"] == workload["backend"] == "original"
    assert result["policy"] == workload["policy"] == "decode_graph"
    assert workload["status"] == "PROFILE_WORKLOAD_CHECKED" and workload["not_timing_sample"]
    assert workload["raw_generated_token_ids"] == plan["raw_expected_token_ids"]
    assert workload["output"][0]["token_ids"] == plan["raw_expected_token_ids"]
    assert len(plan["input_token_ids"]) == 128 and len(workload["raw_generated_token_ids"]) == 32
    assert workload["expected_kv_calls"] == 896
    assert workload["engine_options"]["cuda_graph_bs"] == [1]
    assert workload["packages"] == plan["timing_packages"]
    for path, expected in plan["source_manifest"].items():
        assert sha(run / "source" / path) == expected, path
    src = (run / "m08_profile_worker.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert not (run / "profile.py").exists()
    assert src.index("for _ in range(2):") < src.index("torch.cuda.profiler.start()") < src.index("torch.cuda.nvtx.range_push('M08_ORIGINAL_GENERATE')")
    assert src.index("torch.cuda.nvtx.range_pop()") < src.index("torch.cuda.profiler.stop()")
    for flag in ("--cuda-event-trace=false", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop"):
        assert flag in result["command"]
    assert "summed GPU kernel durations, not wall-clock critical-path fraction" in result["limit"]
    assert "CuptiUseRawGpuTimestamps=false" in (run / "nsys-config.ini").read_text(encoding="utf-8")
    database = run / "original-model.sqlite"
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    aggregate_sql = """SELECT s.value, COUNT(*), SUM(k.end-k.start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id=k.demangledName
        GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC"""
    rows = db.execute(aggregate_sql).fetchall()
    recalculated = {name: {"count": count, "total_ns": ns} for name, count, ns in rows}
    assert recalculated == result["kernels"]
    total_count, total_ns = db.execute("SELECT COUNT(*),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    invalid_durations = db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end<=start").fetchone()[0]
    assert invalid_durations == 0
    kv_sql = """SELECT COUNT(*),SUM(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,'store_kv_cache<')>0"""
    kv_count, kv_ns = db.execute(kv_sql).fetchone()
    share = 100 * kv_ns / total_ns
    assert kv_count == result["kv_count"] == 896
    assert kv_ns == result["kv_total_ns"] == 812245
    assert total_ns == result["all_kernel_total_ns"] == 103927254
    assert share == result["kv_kernel_duration_share_pct"]
    nvtx = db.execute("""SELECT n.start,n.end,coalesce(n.text,s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE coalesce(n.text,s.value)='M08_ORIGINAL_GENERATE'""").fetchall()
    assert len(nvtx) == 1 and nvtx[0][1] > nvtx[0][0]
    range_start, range_end, _ = nvtx[0]
    outside = db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<? OR end>?", (range_start, range_end)).fetchone()[0]
    assert outside == 0
    graph_sql = """SELECT CASE WHEN coalesce(k.graphNodeId,0)>0 THEN 1 ELSE 0 END AS in_graph,
        k.gridX,k.blockX,COUNT(*),SUM(k.end-k.start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
        WHERE instr(s.value,'store_kv_cache<')>0 GROUP BY in_graph,k.gridX,k.blockX"""
    graph_groups = db.execute(graph_sql).fetchall()
    assert {(flag, grid, block, count) for flag, grid, block, count, _ in graph_groups} == {(0, 32, 128, 28), (1, 1, 128, 868)}
    streams = db.execute("SELECT deviceId,contextId,streamId,COUNT(*),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId,contextId,streamId").fetchall()
    db.close()
    csv_lines = (run / "nsys-kernels.csv").read_text(encoding="utf-8").splitlines()
    header = next(i for i, line in enumerate(csv_lines) if line.startswith("Time (%),"))
    csv_rows = list(csv.DictReader(csv_lines[header:]))
    assert {x["Name"]: {"count": int(x["Instances"]), "total_ns": int(x["Total Time (ns)"])} for x in csv_rows} == recalculated
    prior_result = read(previous / "result.json")
    assert prior_result["status"] == "FAILED" and prior_result["profile_returncode"] == 1
    assert (previous / "profile.py").is_file()
    assert "ImportError" in (previous / "nsys-profile.log").read_text(encoding="utf-8")
    assert previous_before == tree_hashes(previous)
    timing_path = run.parent.parent / "m08-model-timing-results" / Path(result["timing_run"]).name
    assert plan == read(timing_path / "plan.json")
    assert read(timing_path / "summary.json")["status"] == "MODEL_TIMING_COLLECTION_COMPLETE"
    return {"status": "INDEPENDENT_CPU_PROFILE_AUDIT_PASS", "gpu_run": False,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": sha(Path(__file__)),
            "run": run.name, "read_only_sqlite": True, "sqlite_integrity": "ok",
            "source_manifest_entries_verified": len(plan["source_manifest"]),
            "saved_profile_worker_matches_current_43": sha(run / "m08_profile_worker.py") == sha(run.parents[2] / "43-m08-model-profile.py"),
            "artifacts_sha256": {p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()},
            "kernel_totals": {"count": total_count, "named_specializations": len(rows), "total_ns": total_ns, "invalid_duration_rows": invalid_durations},
            "kv_totals": {"count": kv_count, "total_ns": kv_ns, "summed_kernel_duration_share_pct": share},
            "top5": [{"name": name, "count": count, "total_ns": ns, "total_ms": ns / 1e6, "summed_kernel_duration_share_pct": 100 * ns / total_ns} for name, count, ns in rows[:5]],
            "nvtx_capture_check": {"range_name": nvtx[0][2], "start_ns": range_start, "end_ns": range_end,
                                   "instrumented_host_range_ns": range_end - range_start, "kernel_rows_outside_range": outside,
                                   "interpretation": "Range coverage check only; duration is not unprofiled generate wall-clock or Amdahl f"},
            "kv_graph_groups": [{"graph_node": bool(flag), "grid_x": grid, "block_x": block, "count": count, "total_ns": ns,
                                  "attribution": "decode graph KV nodes" if flag else "eager prefill KV kernels"} for flag, grid, block, count, ns in graph_groups],
            "kernel_device_context_stream_groups": [list(x) for x in streams],
            "queries": {"all_named_kernels": aggregate_sql, "kv_kernels": kv_sql, "kv_graph_groups": graph_sql},
            "csv_agrees_with_sqlite": True,
            "scope_review": {"two_warmups_before_capture": True, "one_generate_inside_cuda_profiler_range": True,
                             "model_original_unpatched": True, "sampling_and_gpu_completion_in_capture": True,
                             "sum_not_wall_clock_critical_path_claim": True, "ws_timestamp_setting": "CuptiUseRawGpuTimestamps=false",
                             "full_prefill_decode_stage_timing_available": False,
                             "phase_limit": "KV nodes alone distinguish 28 eager prefill vs 868 decode graph calls. Non-graph kernels also run during decode; do not equate all non-graph kernels with prefill."},
            "prior_failed_run": {"path": previous.name, "status": prior_result["status"], "entry": "profile.py",
                                 "recorded_error": "Transformers GenerationMixin ImportError", "files_unchanged": True,
                                 "retained_file_sha256": previous_before},
            "limits": ["One instrumented original workload, with decode Graph node tracing and 128 input/32 generated tokens; no profiled Triton pair.",
                       "The denominator is SUM of GPU kernel durations; it is neither generate wall-clock nor a nonoverlapping critical-path fraction.",
                       "No inference of bandwidth, occupancy, counter-based bottleneck, statistical confidence, or end-to-end speedup from this share.",
                       "WSL timestamps and profiler perturbation limit interpretation; retain unprofiled 42 data separately."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run / "audit.json"
    if output.exists():
        raise FileExistsError("Existing audit preserved")
    result = audit(args.run)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "kernel_totals", "kv_totals", "top5", "kv_graph_groups", "nvtx_capture_check")}, indent=2))
