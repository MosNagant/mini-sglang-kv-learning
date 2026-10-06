"""Run six original-CUDA pilot cases for three rounds, preserving both policies.

Coach-supplied orchestration reuses the reviewed learner timer in script 17.
Each case is a fresh process. Completion is data collection, not M05 acceptance.
"""

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = [(b, d) for b in (1, 16, 256) for d in (64, 128)]
    plan = {
        "cases": [{"batch": b, "width": d} for b, d in cases],
        "capacity": 1024, "dtype": "float16", "indices": "0..batch-1",
        "layout": "interleaved", "input_seed": 20261004,
        "order_seed": 104, "rounds": 3, "pairs_per_case": 30,
        "calls_per_interval": 100, "primary": "graph", "reference": "eager",
        "policy": "same reused tensors; 100 captured calls per graph; fresh process per case/round",
        "kv_pool_bytes_at_max_width": 1024 * 2 * 128 * 2,
        "kv_inputs_bytes_at_max_case": 256 * 128 * 2 * 2,
        "allocation_note": "payload counts exclude expected snapshots, graph resources and runtime",
        "limits": "pilot only; no cold-cache/DRAM, Triton, model or optimization result",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not (Path.cwd() / "experiments/activate.sh").is_file():
        raise RuntimeError("Run from the activated WSL project root.")
    root = Path.cwd() / "experiments/learning/m05-original-pilot"
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    runner = Path(__file__).resolve().with_name("17-compare-event-graph.py")
    # Execute a frozen copy so later edits cannot alter a running matrix.
    frozen = run / "runner.py"
    shutil.copyfile(runner, frozen)
    shutil.copyfile(__file__, run / "driver.py")
    (run / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print("PILOT_DIRECTORY:", run, flush=True)
    all_results = []
    rng = random.Random(plan["order_seed"])
    reference_environment = None
    for round_id in range(1, 4):
        order = cases.copy()
        rng.shuffle(order)
        for b, d in order:
            parent = run / f"round-{round_id}" / f"B{b}-D{d}"
            parent.mkdir(parents=True)
            print(f"RUN round={round_id}/3 B={b} D={d}", flush=True)
            command = [sys.executable, str(frozen), "--batch", str(b), "--width", str(d),
                       "--random-seed", str(plan["input_seed"]), "--result-parent", str(parent)]
            with (parent / "console.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                print((parent / "console.log").read_text(encoding="utf-8"), flush=True)
                raise RuntimeError(f"PILOT_STOPPED; preserve error log: {parent / 'console.log'}")
            paths = list(parent.glob("run-*/result.json"))
            if len(paths) != 1:
                raise RuntimeError(f"Expected one case result, found {paths}")
            data = json.loads(paths[0].read_text(encoding="utf-8"))
            if (data["status"] != "DIAGNOSTIC_COMPLETE" or
                    data["batch"] != b or data["width"] != d or
                    data["capacity"] != 1024 or data["calls_per_sample"] != 100 or
                    data["indices"] != list(range(b)) or data["seed"] != plan["input_seed"]):
                raise RuntimeError(f"Result/plan mismatch: {paths[0]}")
            if data["tracked_git_status"]:
                raise RuntimeError("Tracked source changed; result saved, review before aggregation.")
            environment = tuple(data[x] for x in ("commit", "torch", "cuda", "gpu"))
            if reference_environment is None:
                reference_environment = environment
            if environment != reference_environment:
                raise RuntimeError("Environment changed between cases; inspect saved results.")
            for method in ("graph", "eager"):
                values = data["samples_ms_per_call"][method]
                if len(values) != 30 or not all(math.isfinite(x) and x > 0 for x in values):
                    raise RuntimeError(f"Invalid samples: {paths[0]} {method}")
                all_results.append({"round": round_id, "batch": b, "width": d,
                                    "method": method, "median_us": statistics.median(values) * 1000,
                                    "min_us": min(values) * 1000, "max_us": max(values) * 1000,
                                    "result_file": str(paths[0])})
            print(f"COLLECTED graph={all_results[-2]['median_us']:.6f} us/call; "
                  f"eager={all_results[-1]['median_us']:.6f} us/call", flush=True)
            (run / "partial-summary.json").write_text(
                json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    with (run / "per-round-summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0]))
        writer.writeheader()
        writer.writerows(all_results)
    summary = []
    for b, d in cases:
        row = {"batch": b, "width": d}
        for method in ("graph", "eager"):
            values = [x["median_us"] for x in all_results
                      if x["batch"] == b and x["width"] == d and x["method"] == method]
            med = statistics.median(values)
            row[method] = {"round_medians_us": values, "median_of_round_medians_us": med,
                           "round_median_span_pct": (max(values) - min(values)) / med * 100}
        summary.append(row)
        print(f"SUMMARY B={b} D={d} graph_round_medians_us={row['graph']['round_medians_us']} "
              f"eager_round_medians_us={row['eager']['round_medians_us']}", flush=True)
    (run / "summary.json").write_text(json.dumps(
        {"status": "PILOT_COLLECTION_COMPLETE", "plan": plan, "summary": summary}, indent=2
    ) + "\n", encoding="utf-8")
    print("SAVED:", run / "summary.json", flush=True)
    print("PILOT_COLLECTION_COMPLETE: 6 cases x 3 processes; review stability before baseline acceptance.")


if __name__ == "__main__":
    main()
