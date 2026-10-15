"""Bounded synthetic-SQLite controls for 47 attribution; no CUDA or model run.

The valid fixture deliberately places GPU kernels OUTSIDE the CPU linear NVTX
range. Attribution must follow (process, correlation), then the CPU API thread
and interval. This script edits only temporary fixtures and its test receipt.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile


HERE = Path(__file__).resolve().parent
DEFAULT_ANALYZER = HERE.parent / "47-analyze-linear-attribution.py"
PID = (17 << 24)
TID = PID | 23
LABEL = "M08_LINEAR_1"


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(run):
    run.mkdir()
    calls = [{"label": LABEL, "phase": "decode", "module_path": "layers.0.self_attn.qkv_proj",
              "input": {"shape": [1, 1024]}, "weight": {"shape": [4096, 1024]}}]
    save(run / "linear-calls.json", calls)
    save(run / "workload.json", {"synthetic_cpu_fixture": True})
    save(run / "plan.json", {"synthetic_cpu_fixture": True, "policy": "eager"})
    db = sqlite3.connect(run / "original-model.sqlite")
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER,
                                  globalTid INTEGER, endGlobalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
          (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER
          (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
          (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER,
           demangledName INTEGER, shortName INTEGER, graphNodeId INTEGER,
           streamId INTEGER, deviceId INTEGER, gridX INTEGER, gridY INTEGER,
           gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER);
    """)
    db.executemany("INSERT INTO StringIds VALUES (?,?)", [
        (1, "cudaLaunchKernel"), (2, "cuLaunchKernel"), (3, "internal::gemvx::kernel_test"),
        (4, "void store_kv_cache<test>"), (5, "vectorized_elementwise_copy"),
        (6, "ampere_fp16_s16816gemm_fp16_test"), (7, LABEL), (8, "M08_LINEAR_2"),
    ])
    db.execute("INSERT INTO NVTX_EVENTS VALUES (100,200,?,NULL,?,NULL)", (LABEL, TID))
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", [
        (120, 140, TID, 101, 1), (220, 230, TID, 102, 1)])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1000, 1100, PID, 101, 3, 3, None, 7, 0, 2, 1, 1, 128, 1, 1),
        (1200, 1250, PID, 102, 4, 4, None, 7, 0, 1, 1, 1, 128, 1, 1)])
    db.commit()
    return db


def add_second_call(run, db, start=160, end=250, api_start=170, api_end=180):
    calls = json.loads((run / "linear-calls.json").read_text())
    second = dict(calls[0], label="M08_LINEAR_2", module_path="layers.0.self_attn.o_proj")
    save(run / "linear-calls.json", calls + [second])
    db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL,?,NULL)",
               (start, end, second["label"], TID))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
               (api_start, api_end, TID, 101, 1))


def mutate(name, run, db):
    if name == "registered_nvtx_string":
        db.execute("UPDATE NVTX_EVENTS SET text=NULL,textId=7")
    elif name == "driver_only":
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER SELECT start,end,globalTid,correlationId,2 FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=101")
        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=101")
    elif name == "nested_runtime_driver":
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (125,135,?,101,2)", (TID,))
    elif name == "same_correlation_other_process":
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (125,135,?,101,1)", ((18 << 24) | 23,))
    elif name == "multiple_kernels_one_linear":
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1300,1320,?,101,5,5,NULL,7,0,1,1,1,32,1,1)", (PID,))
    elif name == "wrong_cpu_thread":
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=? WHERE correlationId=101", (PID | 24,))
    elif name == "wrong_kernel_process":
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=? WHERE correlationId=101", (18 << 24,))
    elif name == "missing_nvtx_range":
        db.execute("DELETE FROM NVTX_EVENTS")
    elif name == "api_crosses_nvtx_end":
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET end=201 WHERE correlationId=101")
    elif name == "cross_thread_range_end":
        db.execute("UPDATE NVTX_EVENTS SET endGlobalTid=?", (PID | 24,))
    elif name == "duplicate_label_range":
        db.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS")
    elif name == "overlapping_linear_ranges":
        add_second_call(run, db)
    elif name == "different_labels_one_correlation":
        add_second_call(run, db, 300, 400, 320, 330)
    elif name == "graph_node_in_eager":
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=4294967296 WHERE correlationId=101")
    elif name == "labelled_and_unlabelled_api_collision":
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (300,310,?,101,2)", (PID | 24,))
    elif name == "duplicate_runtime_correlation":
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (125,135,?,101,1)", (TID,))
    elif name == "non_nested_runtime_driver":
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (150,160,?,101,2)", (TID,))
    elif name == "unmatched_gemm_despite_matched_copy":
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=5,shortName=5 WHERE correlationId=101")
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=6,shortName=6 WHERE correlationId=102")
    elif name == "nonpositive_kernel_duration":
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start WHERE correlationId=101")
    elif name != "gpu_time_outside_cpu_range":
        raise ValueError(name)
    db.commit()


def execute(analyzer_path):
    spec = importlib.util.spec_from_file_location("m08_attribution_under_test", analyzer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    positive = ["gpu_time_outside_cpu_range", "registered_nvtx_string", "driver_only",
                "nested_runtime_driver", "same_correlation_other_process", "multiple_kernels_one_linear"]
    negative = ["wrong_cpu_thread", "wrong_kernel_process", "missing_nvtx_range",
                "api_crosses_nvtx_end", "cross_thread_range_end", "duplicate_label_range",
                "overlapping_linear_ranges", "different_labels_one_correlation", "graph_node_in_eager",
                "labelled_and_unlabelled_api_collision", "duplicate_runtime_correlation",
                "non_nested_runtime_driver", "unmatched_gemm_despite_matched_copy",
                "nonpositive_kernel_duration"]
    records = []
    with tempfile.TemporaryDirectory(prefix="m08-attribution-cpu-") as root:
        for name in positive + negative:
            run = Path(root) / name
            db = fixture(run)
            mutate(name, run, db)
            db.close()
            should_pass = name in positive
            try:
                result = module.analyze(run)
            except RuntimeError as error:
                records.append({"case": name, "expected": "accept" if should_pass else "reject",
                                "actual": "reject", "passed": not should_pass, "message": str(error)})
            else:
                wanted_kernels = 3 if name == "multiple_kernels_one_linear" else 2
                wanted_matched = 2 if name == "multiple_kernels_one_linear" else 1
                wanted_total = 170 if name == "multiple_kernels_one_linear" else 150
                wanted_linear_total = 120 if name == "multiple_kernels_one_linear" else 100
                valid = (result["status"] == "LINEAR_KERNEL_ATTRIBUTION_COMPLETE"
                         and result["linear_calls"] == 1 and result["linear_nvtx_ranges"] == 1
                         and result["kernel_count"] == wanted_kernels
                         and result["matched_linear_kernels"] == wanted_matched
                         and result["kv_kernel_count"] == 1
                         and result["kv_kernel_total_ns"] == 50
                         and result["all_kernel_total_ns"] == wanted_total
                         and result["linear_kernel_total_ns"] == wanted_linear_total)
                records.append({"case": name, "expected": "accept" if should_pass else "reject",
                                "actual": "accept", "passed": should_pass and valid,
                                "counts_and_durations_match": valid})
            gc.collect()
    return {"status": "SYNTHETIC_SQLITE_CONTROLS_PASS" if all(r["passed"] for r in records)
                       else "SYNTHETIC_SQLITE_CONTROLS_FAIL",
            "analyzer_path": str(analyzer_path.resolve()),
            "analyzer_sha256": hashlib.sha256(analyzer_path.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "new_gpu_runs": 0, "positive_cases": len(positive), "negative_cases": len(negative),
            "passed_cases": sum(r["passed"] for r in records), "cases": records,
            "limits": "Synthetic CPU fixtures test attribution rules; not real model attribution or performance evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer", type=Path, default=DEFAULT_ANALYZER)
    parser.add_argument("--output", type=Path, default=HERE / "20261015-m08-linear-attribution-cpu-test.json")
    args = parser.parse_args()
    result = execute(args.analyzer)
    save(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "positive_cases", "negative_cases", "passed_cases")}, indent=2))
    if result["status"] != "SYNTHETIC_SQLITE_CONTROLS_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
