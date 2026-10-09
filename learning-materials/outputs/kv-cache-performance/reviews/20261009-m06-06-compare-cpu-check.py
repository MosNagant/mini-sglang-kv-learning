"""Bounded CPU checks of M06-06 schedule, timer reuse, and aggregation APIs.

Synthetic numbers and event/graph stand-ins test formulas and call ordering.
No worker, GPU, CUDA Graph capture, real timing, or performance claim is run.
The accepted timer body is written independently below; temporary archives
include top-level failures to prove only the reviewed function is executed.
"""
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import math
import random
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REVIEWED_TIMER = """
def measure_once(call, start, end):
    start.record()
    for _ in range(100):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100
"""
IMPLS = ("original_cuda", "learner_triton")
POLICIES = ("graph", "eager")
CASES = [(b, d) for b in (1, 16, 256) for d in (64, 128)]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Event:
    def __init__(self, name, log, elapsed_ms=7.5):
        self.name, self.log, self.elapsed_ms = name, log, elapsed_ms

    def record(self):
        self.log.append(self.name + ".record")

    def synchronize(self):
        self.log.append(self.name + ".synchronize")

    def elapsed_time(self, end):
        assert self.name == "start" and end.name == "end"
        self.log.append("start.elapsed_time(end)")
        return self.elapsed_ms


class Graph:
    def __init__(self, log, fail=False):
        self.log, self.fail = log, fail
        self.replays = 0
        self.represented_calls = 0

    def replay(self):
        self.log.append("graph.replay")
        self.replays += 1
        if self.fail:
            raise RuntimeError("CPU_TEST_REPLAY_FAILURE")
        self.represented_calls += 100


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def fixture_results():
    # Graph representatives: original 9, candidate 4 => 2.25x. Paired ratios
    # [0.2,4.5,2.5] have median 2.5, deliberately a different statistic.
    # Eager representatives: original 5, candidate 10 => 0.5x / -100% reduction.
    rounds_us = {
        "graph": {"original_cuda": [1.0, 9.0, 10.0], "learner_triton": [5.0, 2.0, 4.0]},
        "eager": {"original_cuda": [12.0, 3.0, 5.0], "learner_triton": [6.0, 12.0, 10.0]},
    }
    results = []
    for r in (1, 2, 3):
        for case_index, (b, d) in enumerate(CASES):
            scale = case_index + 1
            samples = {}
            for impl in IMPLS:
                samples[impl] = {}
                for policy in POLICIES:
                    center_ms = rounds_us[policy][impl][r-1] * scale / 1000
                    # 30 positive samples: median=center, mean != center.
                    values = [0.8*center_ms]*15 + [1.2*center_ms]*14 + [100*center_ms]
                    samples[impl][policy] = list(reversed(values)) if r % 2 else values
            results.append(dict(status="COLLECTED", round=r, batch=b, width=d,
                                samples_ms_per_call=samples))
    return results, rounds_us


def run():
    module = load(HERE / "25-compare-kv-v1.py", "paired_cpu_module")
    records = []
    expected_combinations = {(i, p) for i in IMPLS for p in POLICIES}
    for r in (1, 2, 3):
        combinations = collections.Counter()
        precedence = {p: collections.Counter() for p in POLICIES}
        policy_precedence = collections.Counter()
        positions = {pair: collections.Counter() for pair in expected_combinations}
        for sample in range(30):
            order = module.measurement_order(sample, r)
            assert len(order) == 4 and set(order) == expected_combinations
            combinations.update(order)
            for position, pair in enumerate(order):
                positions[pair][position] += 1
            for p in POLICIES:
                first = min(IMPLS, key=lambda i: order.index((i,p)))
                precedence[p][first] += 1
            first_policy = min(POLICIES, key=lambda p: min(order.index((i,p)) for i in IMPLS))
            policy_precedence[first_policy] += 1
        assert set(combinations.values()) == {30}
        assert all(counts == {"original_cuda":15, "learner_triton":15} for counts in precedence.values())
        assert policy_precedence == {"graph":15, "eager":15}
        assert all(sorted(counts.values()) == [15,15] for counts in positions.values())
        records.append(dict(name="balanced_measurement_order", round=r, status="CPU_PASS",
                            original_first_per_policy={p:precedence[p]["original_cuda"] for p in POLICIES},
                            graph_first=policy_precedence["graph"], measurements=120))
    global_rng_state = random.getstate()
    plan = module.make_plan()
    assert random.getstate() == global_rng_state
    assert module.make_plan() == plan
    schedule = plan["schedule"]
    assert len(schedule) == 18 and len({(x["round"],x["batch"],x["width"]) for x in schedule}) == 18
    for r in (1, 2, 3):
        assert {(x["batch"],x["width"]) for x in schedule if x["round"] == r} == set(CASES)
    for name, value in dict(capacity=1024, dtype="float16", layout="interleaved", input_gap=0,
                            input_seed=20261004, order_seed=104, rounds=3, intervals_per_impl_policy=30,
                            calls_per_interval=100, primary="graph", reference="eager").items():
        assert plan[name] == value, (name,plan[name],value)
    records.append(dict(name="deterministic_complete_plan",status="CPU_PASS",processes=18,
                        interval_averages=18*4*30, independent_global_rng=True))

    with tempfile.TemporaryDirectory(prefix="m06-06-cpu-timer-") as temp:
        directory = Path(temp)
        path = directory / "archived-timer.py"
        path.write_text("\ufeffraise RuntimeError('TOP_LEVEL_MUST_NOT_EXECUTE')\n" + REVIEWED_TIMER +
                        "\nraise RuntimeError('TRAILING_TOP_LEVEL_MUST_NOT_EXECUTE')\n",encoding="utf-8")
        timer = module.load_reviewed_timer(path)
        log = []
        start, end = Event("start",log), Event("end",log)
        value = timer(lambda: log.append("call"),start,end)
        assert log == ["start.record"] + ["call"]*100 + ["end.record","end.synchronize","start.elapsed_time(end)"]
        close(value,7.5/100)
        records.append(dict(name="reviewed_event_timer_order_and_normalization",status="CPU_PASS",
                            cpu_standin_calls=100,synthetic_interval_ms=7.5,result_ms_per_call=value,
                            archive_top_level_executed=False))
        variants = {
            "loop_99":REVIEWED_TIMER.replace("range(100)","range(99)"),
            "divisor_99":REVIEWED_TIMER.replace(" / 100"," / 99"),
            "missing_end_synchronize":REVIEWED_TIMER.replace("    end.synchronize()\n",""),
            "duplicate_function":REVIEWED_TIMER+REVIEWED_TIMER,
            "missing_function":"def other():\n    pass\n",
            "decorated_function":REVIEWED_TIMER.replace("def measure_once", "@not_allowed\ndef measure_once"),
        }
        for name, source in variants.items():
            bad = directory / (name+".py")
            bad.write_text(source,encoding="utf-8")
            try:
                module.load_reviewed_timer(bad)
            except RuntimeError as error:
                assert str(error).startswith("REVIEWED_TIMER_CHANGED")
                records.append(dict(name="timer_reject_"+name,status="CPU_PASS",error=str(error)))
            else:
                raise AssertionError("Changed timer accepted: "+name)
        log = []
        graph = Graph(log)
        value = module.measure_graph(graph, Event("start",log), Event("end",log))
        assert log == ["start.record","graph.replay","end.record","end.synchronize","start.elapsed_time(end)"]
        assert graph.replays == 1 and graph.represented_calls == 100
        close(value,7.5/100)
        records.append(dict(name="graph_one_replay_divide_100",status="CPU_PASS",replays=1,
                            represented_calls=100,result_ms_per_call=value))
        log = []
        try:
            module.measure_graph(Graph(log,fail=True),Event("start",log),Event("end",log))
        except RuntimeError as error:
            assert str(error) == "CPU_TEST_REPLAY_FAILURE"
        else:
            raise AssertionError("Replay error swallowed")
        assert log == ["start.record","graph.replay"]
        records.append(dict(name="graph_failure_propagates_without_stale_elapsed",status="CPU_PASS"))
        log = []
        def broken_call():
            log.append("call")
            if log.count("call") == 37:
                raise RuntimeError("CPU_TEST_CALL_FAILURE")
        try:
            timer(broken_call,Event("start",log),Event("end",log))
        except RuntimeError as error:
            assert str(error) == "CPU_TEST_CALL_FAILURE"
        else:
            raise AssertionError("Eager error swallowed")
        assert log == ["start.record"] + ["call"]*37
        records.append(dict(name="eager_failure_propagates_without_stale_elapsed",status="CPU_PASS"))

    results, centers = fixture_results()
    pristine = copy.deepcopy(results)
    summary = module.summarize(list(reversed(results)))
    assert results == pristine
    assert len(summary) == 6
    for case_index, row in enumerate(summary):
        assert (row["batch"],row["width"]) == CASES[case_index]
        scale = case_index+1
        for policy in POLICIES:
            output = row[policy]
            representatives = {"graph":{"original_cuda":9.0,"learner_triton":4.0},
                               "eager":{"original_cuda":5.0,"learner_triton":10.0}}[policy]
            for impl in IMPLS:
                expected_medians = [x*scale for x in centers[policy][impl]]
                for actual, expected in zip(output[impl]["round_medians_us"],expected_medians):
                    close(actual,expected)
                assert len(output[impl]["round_medians_us"]) == 3
                representative = representatives[impl]*scale
                close(output[impl]["representative_us"],representative)
                close(output[impl]["round_span_pct"],100*(max(expected_medians)-min(expected_medians))/representative)
            a, b = representatives["original_cuda"],representatives["learner_triton"]
            close(output["speedup_original_over_triton"],a/b)
            close(output["time_reduction_pct"],100*(1-b/a))
            for actual, first, second in zip(output["paired_round_speedups"],centers[policy]["original_cuda"],centers[policy]["learner_triton"]):
                close(actual,first/second)
    close(summary[0]["graph"]["speedup_original_over_triton"],2.25)
    close(summary[0]["eager"]["time_reduction_pct"],-100)
    records.append(dict(name="synthetic_summary_formulas_units_and_order",status="CPU_PASS",worker_results=18,
                        graph_ratio_of_representatives=2.25,graph_median_of_round_ratios_is_different=2.5,
                        eager_negative_reduction_pct=-100,synthetic=True))

    def reject(name, bad, expected_prefix=None, error_type=ValueError):
        try:
            module.summarize(bad)
        except error_type as error:
            if expected_prefix:
                assert str(error).startswith(expected_prefix), (name,str(error))
            records.append(dict(name="summary_reject_"+name,status="CPU_PASS",error_type=type(error).__name__,error=str(error)))
        else:
            raise AssertionError("Invalid matrix accepted: "+name)
    reject("missing_result",copy.deepcopy(results[:-1]),"INCOMPLETE_18_PROCESS_MATRIX")
    reject("duplicate_result",copy.deepcopy(results+[results[0]]),"RESULT_ID_OR_STATUS_MISMATCH")
    for name, key, value in (("failed_status","status","FAILED"),("unexpected_round","round",4),
                              ("unexpected_batch","batch",3),("unexpected_width","width",192)):
        bad = copy.deepcopy(results)
        bad[0][key] = value
        reject(name,bad,"RESULT_ID_OR_STATUS_MISMATCH")
    for name, values in (("29_samples",[0.1]*29),("31_samples",[0.1]*31),
                         ("zero_sample",[0.1]*29+[0.0]),("negative_sample",[0.1]*29+[-0.1]),
                         ("nan_sample",[0.1]*29+[float("nan")]),("infinite_sample",[0.1]*29+[float("inf")])):
        bad = copy.deepcopy(results)
        bad[0]["samples_ms_per_call"]["original_cuda"]["graph"] = values
        reject(name,bad,"INVALID_INTERVAL_SAMPLES")
    bad = copy.deepcopy(results)
    del bad[0]["samples_ms_per_call"]["learner_triton"]
    reject("missing_implementation",bad,error_type=KeyError)
    bad = copy.deepcopy(results)
    del bad[0]["samples_ms_per_call"]["original_cuda"]["eager"]
    reject("missing_policy",bad,error_type=KeyError)
    return dict(status="CPU_COMPARE_API_CHECK_PASS",GPU="NOT_RUN",real_timing="NOT_RUN",
                performance_claim="NONE; all timing values below are synthetic test inputs",
                scope="Pure schedule/order/timer-extraction/aggregation APIs only; worker/capture/GPU behavior is not validated",
                checks=len(records),results=records,
                source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
                               ("25-compare-kv-v1.py","17-compare-event-graph.py","18-original-baseline-pilot.py")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    result = run()
    result["test_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x",encoding="utf-8",newline="\n") as out:
        json.dump(result,out,indent=2)
        out.write("\n")
    print("CPU_COMPARE_API_CHECK_PASS: " + str(result["checks"]) + " bounded checks; GPU NOT_RUN; real timing NOT_RUN")
