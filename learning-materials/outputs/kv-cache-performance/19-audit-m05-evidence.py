"""Coach-only, offline audit of exported M05 pilot artifacts; never runs GPU code.

Exit 0 means saved artifacts are consistent, NOT kernel correctness or M05 acceptance.
Input files are read only. Outputs use exclusive creation in a separate directory.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"
DEFAULT_RUN = "experiments/learning/m05-original-pilot/run-d9gjhywx"
CASES = [(b, d) for b in (1, 16, 256) for d in (64, 128)]
REL_TOL, ABS_TOL = 1e-9, 1e-9
PLAN = {
    "cases": [{"batch": b, "width": d} for b, d in CASES],
    "capacity": 1024, "dtype": "float16", "indices": "0..batch-1",
    "layout": "interleaved", "input_seed": 20261004, "order_seed": 104,
    "rounds": 3, "pairs_per_case": 30, "calls_per_interval": 100,
    "primary": "graph", "reference": "eager",
    "policy": "same reused tensors; 100 captured calls per graph; fresh process per case/round",
    "kv_pool_bytes_at_max_width": 524288, "kv_inputs_bytes_at_max_case": 131072,
    "allocation_note": "payload counts exclude expected snapshots, graph resources and runtime",
    "limits": "pilot only; no cold-cache/DRAM, Triton, model or optimization result",
}
TIMER = """
def measure_once(call, start, end):
    start.record()
    for _ in range(100):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100
"""
# Optional provenance comparison with the reviewed coach files as of 2026-10-06.
# Differences are reported as UNKNOWN, not silently accepted or treated as bad GPU data.
REFERENCE_AST = {
    "runner.py": "3587354009a9d25b98f7c53cff8435187263c382024ac15fb9fd7ffca7e35a8c",
    "driver.py": "020506e3f4a378e03d191231f3a7d8031ed03bcc6bed5e84d4198b19cfb7adbe",
}


def long_path(value):
    value = os.path.abspath(os.fspath(value))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value).resolve()


def canonical(node):
    if isinstance(node, ast.AST):
        return {"_type": type(node).__name__, **{
            k: canonical(v) for k, v in ast.iter_fields(node) if v is not None and v != []}}
    return [canonical(v) for v in node] if isinstance(node, list) else node


def ast_hash(text):
    data = json.dumps(canonical(ast.parse(text)), ensure_ascii=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def report_value(value):
    """Keep rejected nonfinite JSON values visible without writing invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: report_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [report_value(v) for v in value]
    return value


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class Invalid(Exception):
    pass


class Audit:
    def __init__(self, run):
        self.run, self.errors, self.unknowns, self.artifacts = run, [], [], {}
        self.rows, self.environment, self.learner_hash = [], None, None
        self.valid_cases = 0

    def require(self, condition, path, field, expected, actual):
        if not condition:
            self.errors.append({"path": str(path), "field": field,
                                "expected": expected, "actual": actual})
            raise Invalid()

    def equal(self, actual, expected, path, field):
        # JSON spelling equality also rejects True==1 and 1.0==1 metadata mistakes.
        same = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        self.require(same, path, field, expected, actual)

    def near(self, actual, expected, path, field):
        ok = finite_number(actual)
        self.require(ok and math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                     path, field, expected, actual)

    def read(self, path):
        self.require(path.resolve().is_relative_to(self.run), path, "path", "inside input run", str(path))
        self.require(path.is_file() and not path.is_symlink(), path, "file", "regular file exists", "missing or symlink")
        self.require(path.stat().st_size <= 16 * 1024 * 1024, path, "size", "<=16 MiB", path.stat().st_size)
        raw = path.read_bytes()
        self.artifacts[path.relative_to(self.run).as_posix()] = {
            "path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return raw

    def json(self, path):
        try:
            return json.loads(self.read(path).decode("utf-8-sig"))
        except (UnicodeError, ValueError) as exc:
            self.require(False, path, "json", "valid UTF-8 JSON", str(exc))

    def timer(self, source, path, field):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            self.require(False, path, field, "parseable Python", str(exc))
        functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == "measure_once"]
        self.require(len(functions) == 1, path, field, "one top-level measure_once", len(functions))
        self.require(canonical(functions[0]) == canonical(ast.parse(TIMER).body[0]),
                     path, field, "reviewed 100-call Event timer AST", ast.dump(functions[0]))

    def source(self, name):
        path = self.run / name
        raw = self.read(path)
        try:
            fingerprint = ast_hash(raw.decode("utf-8-sig"))
        except (UnicodeError, SyntaxError, ValueError) as exc:
            self.require(False, path, "source", "parseable Python", str(exc))
        self.artifacts[name]["ast_sha256"] = fingerprint
        if fingerprint != REFERENCE_AST[name]:
            self.unknowns.append({"path": str(path), "field": "reviewed_coach_source_identity",
                                  "expected_ast_sha256": REFERENCE_AST[name], "actual_ast_sha256": fingerprint})
        return raw

    def case(self, round_id, batch, width, runner):
        parent = self.run / f"round-{round_id}" / f"B{batch}-D{width}"
        runs = sorted(parent.glob("run-*"))
        self.require(len(runs) == 1 and runs[0].is_dir(), parent, "case_runs", "one run-* directory", [p.name for p in runs])
        for directory in (parent.parent, parent, runs[0]):
            linked = directory.is_symlink() or getattr(directory, "is_junction", lambda: False)()
            self.require(not linked, directory, "directory", "ordinary exported directory", "link or junction")
        path = runs[0] / "result.json"
        data = self.json(path)
        self.require(isinstance(data, dict), path, "$", "object", type(data).__name__)
        expected = {"status": "DIAGNOSTIC_COMPLETE", "scope": "method diagnostic, not final baseline",
                    "commit": COMMIT, "tracked_git_status": "", "batch": batch, "width": width,
                    "capacity": 1024, "dtype": "torch.float16", "layout": "interleaved",
                    "input_stride": [width, 1], "cache_stride": [2 * width, 1],
                    "indices": list(range(batch)), "seed": 20261004, "calls_per_sample": 100,
                    "input_pattern": "cpu_seeded_integer_K_0_1023_V_plus_1024"}
        for key, value in expected.items():
            self.equal(data.get(key), value, path, key)
        env = {k: data.get(k) for k in ("commit", "torch", "cuda", "gpu")}
        self.require(all(isinstance(v, str) and v.strip() for v in env.values()), path, "environment", "nonempty strings", env)
        if self.environment is None:
            self.environment = env
        self.equal(env, self.environment, path, "environment")
        diagnostic = self.read(runs[0] / "diagnostic.py")
        self.require(runner is not None and diagnostic == runner, runs[0] / "diagnostic.py",
                     "bytes", "byte-identical to runner.py", hashlib.sha256(diagnostic).hexdigest())
        learner = self.read(runs[0] / "learner_source.py")
        self.timer(learner.decode("utf-8-sig"), runs[0] / "learner_source.py", "measure_once")
        self.timer(data.get("reviewed_measure_once", ""), path, "reviewed_measure_once")
        digest = hashlib.sha256(learner).hexdigest()
        if self.learner_hash is None:
            self.learner_hash = digest
        self.equal(digest, self.learner_hash, runs[0] / "learner_source.py", "same_learner_source_sha256")
        samples = data.get("samples_ms_per_call")
        self.require(isinstance(samples, dict) and set(samples) == {"graph", "eager"}, path,
                     "samples_ms_per_call", "graph and eager only", list(samples) if isinstance(samples, dict) else None)
        rows = []
        for method in ("graph", "eager"):
            values = samples[method]
            valid = isinstance(values, list) and len(values) == 30 and all(
                finite_number(x) and x > 0 for x in values)
            self.require(valid, path, f"samples_ms_per_call.{method}", "30 positive finite numeric ms/call values", values)
            stats = {"median": statistics.median(values) * 1000, "min": min(values) * 1000,
                     "max": max(values) * 1000}
            saved = data.get("summary_us_per_call", {}).get(method, {})
            for key, value in stats.items():
                self.near(saved.get(key), value, path, f"summary_us_per_call.{method}.{key}")
            rows.append({"round": round_id, "batch": batch, "width": width, "method": method,
                         **{key + "_us": value for key, value in stats.items()},
                         "result_file": path.relative_to(self.run).as_posix(),
                         "samples_field": f"samples_ms_per_call.{method}",
                         "result_sha256": self.artifacts[path.relative_to(self.run).as_posix()]["sha256"]})
        self.rows.extend(rows)
        self.valid_cases += 1

    def execute(self):
        plan, summary, runner = None, None, None
        # Each independent artifact/case reports a failure without hiding later missing files.
        for name in ("plan.json", "summary.json", "runner.py", "driver.py"):
            try:
                if name == "plan.json":
                    plan = self.json(self.run / name)
                    self.equal(plan, PLAN, self.run / name, "$")
                elif name == "summary.json":
                    summary = self.json(self.run / name)
                    self.equal(summary.get("status"), "PILOT_COLLECTION_COMPLETE", self.run / name, "status")
                    self.equal(summary.get("plan"), PLAN, self.run / name, "plan")
                else:
                    raw = self.source(name)
                    if name == "runner.py":
                        runner = raw
            except Invalid:
                pass
            except (OSError, UnicodeError, TypeError, AttributeError) as exc:
                self.errors.append({"path": str(self.run / name), "field": "read/schema", "actual": str(exc)})
        expected_results = 18
        try:
            rounds = sorted(p.name for p in self.run.glob("round-*") if p.is_dir())
            self.equal(rounds, ["round-1", "round-2", "round-3"], self.run, "round_directories")
            for r in range(1, 4):
                directories = sorted(p.name for p in (self.run / f"round-{r}").glob("B*") if p.is_dir())
                self.equal(directories, sorted(f"B{b}-D{d}" for b, d in CASES), self.run / f"round-{r}", "case_directories")
            self.equal(len(list(self.run.glob("round-*/B*/run-*/result.json"))), expected_results,
                       self.run, "result_file_count")
        except Invalid:
            pass
        for r in range(1, 4):
            for b, d in CASES:
                try:
                    self.case(r, b, d, runner)
                except Invalid:
                    pass
                except (OSError, UnicodeError, TypeError, AttributeError, ValueError) as exc:
                    self.errors.append({"path": f"round-{r}/B{b}-D{d}", "field": "read/schema", "actual": str(exc)})
        if self.valid_cases == 18 and isinstance(summary, dict):
            try:
                saved_rows = summary.get("summary")
                self.require(isinstance(saved_rows, list) and len(saved_rows) == 6, self.run / "summary.json",
                             "summary", "six aggregate rows", saved_rows)
                self.equal([(x.get("batch"), x.get("width")) for x in saved_rows], CASES,
                           self.run / "summary.json", "summary case order")
                for saved, (b, d) in zip(saved_rows, CASES):
                    for method in ("graph", "eager"):
                        values = [x["median_us"] for x in self.rows if (x["batch"], x["width"], x["method"]) == (b, d, method)]
                        med = statistics.median(values)
                        target = saved.get(method, {})
                        field = f"summary[B{b},D{d}].{method}"
                        existing = target.get("round_medians_us")
                        self.require(isinstance(existing, list) and len(existing) == 3, self.run / "summary.json", field, "three round medians", existing)
                        for i, value in enumerate(values):
                            self.near(existing[i], value, self.run / "summary.json", f"{field}.round_medians_us[{i}]")
                        self.near(target.get("median_of_round_medians_us"), med, self.run / "summary.json", field + ".median_of_round_medians_us")
                        self.near(target.get("round_median_span_pct"), (max(values) - min(values)) / med * 100,
                                  self.run / "summary.json", field + ".round_median_span_pct")
            except Invalid:
                pass
            except (TypeError, AttributeError, ValueError) as exc:
                self.errors.append({"path": "summary.json", "field": "schema", "actual": str(exc)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", help="exported project root (containing experiments/)")
    source.add_argument("--run", help="explicit exported pilot run directory")
    parser.add_argument("--output-dir", required=True, help="separate Windows/workspace directory; must be outside input")
    args = parser.parse_args(argv)
    input_root = long_path(args.source_dir or args.run)
    run = input_root / DEFAULT_RUN if args.source_dir else input_root
    output = long_path(args.output_dir)
    if output.is_relative_to(input_root):
        parser.error("--output-dir must be outside the read-only input tree")
    if not input_root.is_dir() or not run.is_dir():
        parser.error(f"exported input/run directory does not exist: {run}")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit = Audit(run)
    audit.execute()
    status = "ARTIFACT_AUDIT_FAIL" if audit.errors else "ARTIFACT_AUDIT_UNKNOWN" if audit.unknowns else "ARTIFACT_AUDIT_PASS"
    report = {
        "schema_version": 1, "created_utc": stamp, "status": status,
        "input_run": str(run), "pinned_commit": COMMIT, "valid_cases": audit.valid_cases,
        "expected_cases": 18, "numeric_tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "scope": "Offline consistency audit of saved artifacts only; no imported/executed archive code, WSL or GPU.",
        "kernel_correctness_independently_verified": False, "full_m05_acceptance": False,
        "limitations": [
            "Only saved K dtype, input/cache strides and B/D/C metadata are checked; full K/V arrays, shapes, V/index dtype and V stride were not persisted.",
            "Matching commit/torch/cuda/GPU model strings does not establish GPU UUID, clocks, temperature or current machine state.",
            "No trace, kernel execution or runtime assertions were independently checked; Python optimization/assertion mode was not recorded.",
            "Source AST reference assumes the reviewed 17/18 coach scripts. Different ASTs remain UNKNOWN and need review, not automatic baseline rejection/approval.",
            "Original console, partial-summary.json and per-round-summary.csv are not acceptance inputs; output CSV is recalculated from the raw per-case sample arrays.",
            "Pilot scope remains six reused-address configurations; this does not complete M05, demonstrate Triton speedup or measure DRAM traffic.",
        ],
        "reference_ast_sha256": REFERENCE_AST, "environment_as_recorded": audit.environment,
        "artifacts": audit.artifacts, "recomputed_per_round": audit.rows,
        "errors": audit.errors, "unknowns": audit.unknowns,
    }
    json_path, csv_path = output / f"m05-audit-{stamp}.json", output / f"m05-audit-{stamp}.csv"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report_value(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    fields = ["round", "batch", "width", "method", "median_us", "min_us", "max_us",
              "result_file", "samples_field", "result_sha256"]
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit.rows)
    print(f"{status}: {audit.valid_cases}/18 cases; {len(audit.errors)} errors; {len(audit.unknowns)} unknowns")
    print(f"JSON: {json_path}\nCSV: {csv_path}")
    for error in audit.errors[:5]:
        print(f"ERROR: {error['path']} :: {error['field']} :: expected {str(error.get('expected', 'valid schema'))[:160]}")
    return 1 if audit.errors else 2 if audit.unknowns else 0


if __name__ == "__main__":
    sys.exit(main())
