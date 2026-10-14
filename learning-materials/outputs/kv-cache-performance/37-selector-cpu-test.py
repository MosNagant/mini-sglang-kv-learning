"""CPU controls for frozen37 selection, independent fixtures and collection math.

No WSL/GPU, imports of Torch/Triton, or modification of archived evidence.
"""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("selector37_runner_test", HERE / "37-validate-kv-selector.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
s = r.s


def main():
    policy = s.load_policy()
    rejected = []
    def rejection(label, action):
        try:
            action()
        except (ValueError, RuntimeError, KeyError, FileNotFoundError):
            rejected.append(label)
        else:
            raise AssertionError("FAILED_NEGATIVE_CONTROL:" + label)
    assert s.meets_promotion([100] * 3, [97] * 3)
    assert not s.meets_promotion([100] * 3, [100 / 1.03] * 3)
    assert not s.meets_promotion([100] * 3, [96, 96, 97.001])
    cfg = r.configuration(7, 64)
    ctx = s.context_for_config(policy, cfg, "graph")
    assert s.select(policy, ctx)["selected"] == "original_cuda"
    for request in policy["research_requests"]:
        decision = s.select(policy, ctx, request)
        assert decision["selected"] == request and decision["reason"] == "explicit_research_not_promoted"
        assert decision["automatic_promotion"] is False and decision["performance_claim"] is False
    for key, value in [("dtype", "bfloat16"), ("index_dtype", "int64"), ("execution_policy", "unknown"),
                       ("address_policy", "flushed"), ("device", "cpu"), ("layout", "unknown"),
                       ("input_gap", 1), ("cache_guard", 64), ("width", 65), ("capacity", 16),
                       ("batch", 0), ("batch", 513), ("batch", True), ("metadata_and_aliases_validated", False),
                       ("unique_in_range_indices_validated", False), ("k_stride", [128, 1]),
                       ("v_shape", [3, 64]), ("indices_shape", [8]), ("indices_stride", [2]),
                       ("k_cache_stride", [64, 1])]:
        changed = deepcopy(ctx); changed[key] = value
        rejection("context_" + key + "_" + str(value), lambda changed=changed: s.select(policy, changed))
    for field in policy["environment"]:
        changed = deepcopy(ctx); changed["environment"][field] = "changed"
        rejection("environment_" + field, lambda changed=changed: s.select(policy, changed))
    changed = deepcopy(ctx); changed["source_hashes"]["triton_store_grouped.py"] = "0" * 64
    rejection("source_hash", lambda: s.select(policy, changed))
    changed = deepcopy(ctx); del changed["k_cache_stride"]
    rejection("missing_metadata", lambda: s.select(policy, changed))
    changed = deepcopy(ctx); changed["unreviewed_hint"] = "prefer_triton"
    rejection("extra_metadata", lambda: s.select(policy, changed))
    changed_policy = deepcopy(policy); changed_policy["promoted_candidates"] = ["full_groups"]
    rejection("forced_promotion", lambda: s.select(changed_policy, ctx))
    changed_policy = deepcopy(policy); changed_policy["promotion_rule"]["per_round_max_candidate_latency_ratio"] = .99
    rejection("threshold_change", lambda: s.select(changed_policy, ctx))
    rejection("unknown_request", lambda: s.select(policy, ctx, "fastest"))
    rejection("invalid_history_value", lambda: s.meets_promotion([1, 1, 1], [1, float("nan"), 1]))
    fixture_arrays = fixture_elements = 0
    for b, d in r.CASES:
        cfg = r.configuration(b, d); before = r.fixture(cfg)
        correct = {n: before["expected_cache" if n == "cache" else n].copy() for n in ("cache", "k", "v", "indices")}
        r.check_cpu_arrays(cfg, before, correct)
        fixture_arrays += len(before); fixture_elements += sum(x.size for x in before.values())
        assert len(set(before["indices"].tolist())) == b and np.all(before["indices"] >= 0) and np.all(before["indices"] < 512)
        assert np.array_equal(before["k"][:, 0], np.arange(b, dtype=np.float16))
        for label, corrupt in [("target", lambda a: r.cache_views(a["cache"], cfg["layout"])[0].__setitem__((cfg["rows"][0], 0), -3)),
                               ("readonly_k", lambda a: a["k"].__setitem__((0, 0), -3)),
                               ("indices", lambda a: a["indices"].__setitem__(0, 0))]:
            bad = deepcopy(correct); corrupt(bad)
            rejection(f"B{b}_" + label, lambda bad=bad: r.check_cpu_arrays(cfg, before, bad))
        untouched = next(i for i in range(512) if i not in cfg["rows"])
        bad = deepcopy(correct); r.cache_views(bad["cache"], cfg["layout"])[1][untouched, 0] = -3
        rejection(f"B{b}_untouched", lambda: r.check_cpu_arrays(cfg, before, bad))
        if cfg["gap"]:
            bad = deepcopy(correct); bad["v"][0, d] = 0
            rejection(f"B{b}_gap", lambda: r.check_cpu_arrays(cfg, before, bad))
        if cfg["guard"]:
            bad = deepcopy(correct); r.cache_views(bad["cache"], cfg["layout"])[0][0, d] = 0
            rejection(f"B{b}_guard", lambda: r.check_cpu_arrays(cfg, before, bad))
        for mode in r.POLICIES:
            assert s.select(policy, s.context_for_config(policy, cfg, mode))["selected"] == "original_cuda"
    gate = r.historical_gate(HERE, policy)
    assert gate["workers"] == 48 and gate["promoted"] == 0
    results = []
    for rr in (1, 2, 3):
        for b, d in r.CASES:
            cfg = r.configuration(b, d)
            result = dict(status="COLLECTED", round=rr, batch=b, width=d, config=cfg, calls_per_interval=100,
                          correctness_checks=127, policy=r.make_plan(), selection_contexts={}, decisions={},
                          actual_entries={"original_cuda": dict(entry="upstream store_cache", initial_calls=1),
                                          "selected": dict(entry="upstream store_cache", initial_calls=1, same_callable_as_original=True)},
                          samples_ms_per_call={a: {p: [] for p in r.POLICIES} for a in r.ARMS}, measurements=[])
            for mode in r.POLICIES:
                context = s.context_for_config(policy, cfg, mode)
                result["selection_contexts"][mode] = context; result["decisions"][mode] = s.select(policy, context)
            positions = Counter()
            for sample in range(30):
                for position, (arm, mode) in enumerate(r.measurement_order(sample, rr)):
                    positions[(arm, mode, position)] += 1
                    ms = ([12, 1, 13] if arm == "original_cuda" else [10, 2, 14])[rr - 1] / 1000
                    if mode == "eager": ms *= 4 if arm == "original_cuda" else 8
                    result["samples_ms_per_call"][arm][mode].append(ms)
                    result["measurements"].append(dict(sample=sample, arm=arm, policy=mode, ms_per_call=ms))
            assert len(positions) == 16 and set(positions.values()) == {7, 8}
            results.append(result)
    summary = r.summarize(results, policy)
    assert len(summary) == 3
    for row in summary:
        assert row["graph"]["original_over_selected_representative"] == 1.2
        assert row["graph"]["median_paired_ratio"] < 1
        assert row["eager"]["original_over_selected_representative"] == .6
        assert row["interpretation"] == "A/A variation; not candidate speedup"
    rejection("missing_worker", lambda: r.summarize(results[:-1], policy))
    rejection("duplicate_worker", lambda: r.summarize(results + [results[0]], policy))
    for label, mutate in [
        ("failed_worker", lambda x: x[0].update(status="FAILED")),
        ("wrong_round", lambda x: x[0].update(round=4)),
        ("wrong_calls", lambda x: x[0].update(calls_per_interval=1)),
        ("wrong_checks", lambda x: x[0].update(correctness_checks=126)),
        ("wrong_gap", lambda x: x[0]["config"].update(gap=64)),
        ("forced_selected_triton", lambda x: x[0]["decisions"]["graph"].update(selected="full_groups")),
        ("changed_actual_entry", lambda x: x[0]["actual_entries"]["selected"].update(entry="triton")),
        ("changed_launch_identity", lambda x: x[0]["actual_entries"]["selected"].update(same_callable_as_original=False)),
        ("changed_context", lambda x: x[0]["selection_contexts"]["graph"].update(dtype="bfloat16")),
        ("short_samples", lambda x: x[0]["samples_ms_per_call"]["selected"]["graph"].pop()),
        ("nan_sample", lambda x: x[0]["samples_ms_per_call"]["selected"]["graph"].__setitem__(0, float("nan"))),
        ("zero_sample", lambda x: x[0]["samples_ms_per_call"]["selected"]["graph"].__setitem__(0, 0)),
        ("raw_sample_disagreement", lambda x: x[0]["measurements"][0].update(ms_per_call=0.9)),
        ("wrong_order", lambda x: x[0]["measurements"].reverse()),
    ]:
        damaged = deepcopy(results); mutate(damaged)
        rejection(label, lambda damaged=damaged: r.summarize(damaged, policy))
    with tempfile.TemporaryDirectory(prefix="kv37-cpu-") as tmp:
        root = Path(tmp); project = root / "project"; run = root / "snapshot"; run.mkdir()
        archive = HERE / "logs" / r.ACCEPTED35
        for name in r.read(archive / "source_hashes.json"):
            if name.startswith("original_source/"):
                dest = project / name.removeprefix("original_source/")
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(archive / name, dest)
        r.snapshot(project, run)
        r.verify_snapshot(run, project)
        victim = run / "25-compare-kv-v1.py"
        victim.write_bytes(victim.read_bytes() + b"\n# changed test copy\n")
        rejection("snapshot_helper_changed", lambda: r.verify_snapshot(run, project))
    assert "torch" not in sys.modules and "triton" not in sys.modules
    print(json.dumps(dict(status="CPU_CONTROLS_PASS", no_gpu_runtime_imported=True, historical_gate=gate,
                         fixture_arrays=fixture_arrays, fixture_elements=fixture_elements,
                         synthetic_workers=9, synthetic_intervals=1080,
                         threshold_boundary="latency_ratio<=0.97; speedup1.03 alone rejected",
                         negative_controls=len(rejected), rejected=rejected), indent=2))


if __name__ == "__main__":
    main()
