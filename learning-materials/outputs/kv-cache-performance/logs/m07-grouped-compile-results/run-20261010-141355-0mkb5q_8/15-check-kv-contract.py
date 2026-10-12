"""M04 original-CUDA contract suite. Coach harness; preserves learner sources.

48 nonempty GPU cases + 4 empty adapter cases. No timing or Triton claim.
CPU self-test validates fixtures/oracle/negative controls, never GPU kernels.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cases():
    for width in (64, 128):
        for layout in ("interleaved", "separate"):
            for gap in (0, 64):
                for batch in (1, 2, 3, 4, 5, 17):
                    rows = ([31, 0] + list(range(1, batch - 1)))[:batch]
                    yield dict(width=width, layout=layout, gap=gap, batch=batch,
                               capacity=32, rows=rows)
            yield dict(width=width, layout=layout, gap=0, batch=0, capacity=32, rows=[])


def validate(spec):
    b, d, c = spec["batch"], spec["width"], spec["capacity"]
    rows = spec["rows"]
    if b < 0 or c <= 0 or b > c:
        raise ValueError("BATCH_CAPACITY")
    if d not in (64, 128) or spec["gap"] not in (0, 64):
        raise ValueError("WIDTH_OR_GAP_OUT_OF_SCOPE")
    if spec["layout"] not in ("interleaved", "separate"):
        raise ValueError("LAYOUT_OUT_OF_SCOPE")
    if len(rows) != b:
        raise ValueError("INDEX_COUNT")
    if any(type(pos) is not int or not 0 <= pos < c for pos in rows):
        raise ValueError("INDEX_RANGE_OR_TYPE")
    if len(set(rows)) != b:
        raise ValueError("DUPLICATE_INDEX")


def views(cache, layout):
    return (cache[:, 0, :], cache[:, 1, :]) if layout == "interleaved" else (cache[0], cache[1])


def fixture(spec):
    validate(spec)
    b, d, c = spec["batch"], spec["width"], spec["capacity"]
    shape = (c, 2, d) if spec["layout"] == "interleaved" else (2, c, d)
    cache = np.full(shape, -1, dtype=np.float16)
    # Bounded, exactly representable finite inputs; padded storage is a guard.
    k = np.full((b, d + spec["gap"]), -9, dtype=np.float16)
    v = k.copy()
    data = (np.arange(b * d).reshape(b, d) % 97).astype(np.float16)
    k[:, :d] = data + 10
    v[:, :d] = data + 1000
    indices = np.array(spec["rows"], dtype=np.int32)
    expected = cache.copy()
    ek, ev = views(expected, spec["layout"])
    # Independent CPU reference, built BEFORE any GPU call.
    ek[indices] = k[:, :d]
    ev[indices] = v[:, :d]
    return dict(cache=cache, k=k, v=v, indices=indices, expected=expected)


def verify(spec, before, after, learner=None):
    ak, av = views(after["cache"], spec["layout"])
    d = spec["width"]
    if learner is not None:
        learner(ak, av, before["k"][:, :d], before["v"][:, :d], spec["rows"])
    # Whole backing storage checks include untouched slots and input padding.
    for name, expected in (("cache", before["expected"]), ("k", before["k"]),
                           ("v", before["v"]), ("indices", before["indices"])):
        np.testing.assert_array_equal(after[name], expected, err_msg=name)


def valid_after(before):
    return {name: before["expected" if name == "cache" else name].copy()
            for name in ("cache", "k", "v", "indices")}


def controls(learner=None):
    spec = dict(batch=3, width=64, capacity=32, gap=64,
                layout="interleaved", rows=[31, 0, 3])
    before = fixture(spec)
    verify(spec, before, valid_after(before), learner)
    caught = []
    for kind in ("wrong_row", "missing_K_last", "missing_V_last", "swap_KV",
                 "untouched_K", "untouched_V", "input_K", "input_V_padding", "indices"):
        after = valid_after(before)
        k, v = views(after["cache"], spec["layout"])
        if kind == "wrong_row":
            k[[31, 0]] = k[[0, 31]]
        elif kind == "missing_K_last":
            k[31, -1] = -1
        elif kind == "missing_V_last":
            v[3, -1] = -1
        elif kind == "swap_KV":
            k[31], v[31] = v[31].copy(), k[31].copy()
        elif kind == "untouched_K":
            k[2, 0] = 99
        elif kind == "untouched_V":
            v[2, 0] = 99
        elif kind == "input_K":
            after["k"][0, 0] = -99
        elif kind == "input_V_padding":
            after["v"][0, -1] = -99
        else:
            after["indices"][0] = 2
        try:
            verify(spec, before, after, learner)
        except AssertionError:
            caught.append(kind)
        else:
            raise AssertionError("CHECKER_MISSED: " + kind)
    rejected = []
    for name, update in (
        ("negative_index", {"rows": [-1, 0, 3]}),
        ("upper_bound", {"rows": [32, 0, 3]}),
        ("duplicate", {"rows": [3, 0, 3]}),
        ("count", {"rows": [31, 0]}),
        ("capacity", {"capacity": 2}),
        ("unsupported_width", {"width": 96}),
    ):
        try:
            fixture(dict(spec, **update))
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("CONTRACT_ACCEPTED: " + name)
    return dict(positive="PASS", rejected_corruptions=caught, rejected_contracts=rejected,
                invalid_cases_submitted_to_GPU=False)


def cpu_test(output):
    result = controls()
    for spec in cases():
        data = fixture(spec)
        verify(spec, data, valid_after(data))
        k, _ = views(data["cache"], spec["layout"])
        assert k.strides[0] // 2 == spec["width"] * (2 if spec["layout"] == "interleaved" else 1)
        if spec["batch"]:
            assert data["k"].strides[0] // 2 == spec["width"] + spec["gap"]
    result.update(status="CPU_HARNESS_PASS", fixtures=52, GPU="NOT_RUN", learner="NOT_RUN")
    save(output, result)
    print(json.dumps(result))


def execute_case(torch, original, learner, spec, directory):
    directory.mkdir()
    before = fixture(spec)
    np.savez_compressed(directory / "before.npz", **before)
    buffers = {name: torch.from_numpy(before[name].copy()).cuda()
               for name in ("cache", "k", "v", "indices")}
    kc, vc = views(buffers["cache"], spec["layout"])
    d, b = spec["width"], spec["batch"]
    k, v = buffers["k"][:, :d], buffers["v"][:, :d]
    metadata = dict(spec, input_stride=list(k.stride()), cache_stride=list(kc.stride()),
                    dtype="float16", index_dtype="int32", original_calls=0)
    save(directory / "case.json", metadata)
    try:
        if b:
            assert k.stride() == v.stride() == (d + spec["gap"], 1)
            assert kc.stride() == vc.stride() == (d * (2 if spec["layout"] == "interleaved" else 1), 1)
            original(kc, vc, buffers["indices"], k, v)
            metadata["original_calls"] = 1
            torch.cuda.synchronize()
        # Empty input is an explicit no-op in THIS adapter, not an upstream claim.
        after = {name: value.cpu().numpy().copy() for name, value in buffers.items()}
        np.savez_compressed(directory / "raw.npz",
                            **{"before_" + key: val for key, val in before.items()},
                            **{"after_" + key: val for key, val in after.items()})
        verify(spec, before, after, learner)
        metadata["status"] = "PASS" if b else "EMPTY_ADAPTER_PASS"
    except Exception:
        metadata["status"] = "FAILED"
        raise
    finally:
        save(directory / "case.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-self-test", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED: do not use -O")
    if args.cpu_self_test:
        cpu_test(args.cpu_self_test)
        return
    if args.project is None or args.shared_results is None:
        parser.error("--project and --shared-results are required for GPU validation")
    project = args.project.resolve()
    def git(*words):
        return subprocess.check_output(["git", "-C", str(project), *words], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain", "--", "python/minisgl/kernel"):
        raise RuntimeError("PINNED_ORIGINAL_CHANGED: no checkout modified")
    parent = project / "experiments/learning/m04-contract"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = dict(status="PREPARING", commit=COMMIT, cases=[], scope="original CUDA M04 correctness; no timing; no Triton run")
    print("RUN_DIRECTORY=" + str(run), flush=True)
    try:
        shutil.copyfile(__file__, run / "coach_runner.py")
        shutil.copyfile(Path(__file__).with_suffix(".sh"), run / "coach_entry.sh")
        shutil.copyfile(project / "experiments/check_kv_targets.py", run / "check_kv_targets.py")
        for relative in git("ls-files", "--", "python/minisgl/kernel").splitlines():
            destination = run / "original_source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, destination)
        save(run / "source_hashes.json", {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in run.rglob("*") if p.is_file()})
        import torch
        from minisgl.kernel.store import store_cache
        if Path(inspect.getsourcefile(store_cache)).resolve() != project / "python/minisgl/kernel/store.py":
            raise RuntimeError("WRONG_MINISGL_IMPORT")
        spec = importlib.util.spec_from_file_location("saved_learner_check", run / "check_kv_targets.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def learner(kc, vc, k, v, rows):
            module.check_kv_targets(*(torch.from_numpy(a.copy()) for a in (kc, vc, k, v)), rows)
        # Directly test the learner checker as well as the whole-state coach oracle.
        def learner_only(kc, vc, k, v, rows):
            learner(kc, vc, k, v, rows)
        positive = fixture(dict(batch=3, width=64, capacity=32, gap=0, layout="interleaved", rows=[31, 0, 3]))
        kc, vc = views(positive["expected"], "interleaved")
        learner_only(kc, vc, positive["k"], positive["v"], [31, 0, 3])
        for kind in ("K", "V"):
            badk, badv = kc.copy(), vc.copy()
            (badk if kind == "K" else badv)[31, -1] = -99
            try:
                learner_only(badk, badv, positive["k"], positive["v"], [31, 0, 3])
            except AssertionError:
                pass
            else:
                raise AssertionError("LEARNER_CHECK_MISSED_" + kind)
        save(run / "controls.json", controls(learner))
        state.update(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                     gpu=torch.cuda.get_device_name(0), learner_K_V_controls="PASS")
        print("CONTROLS_PASS: learner K/V checks; 9 suite corruptions; 6 invalid configurations rejected before GPU", flush=True)
        for index, case in enumerate(cases()):
            state.update(status="RUNNING", current_case=case)
            save(run / "status.json", state)
            result = execute_case(torch, store_cache, learner, case, run / f"case-{index:02d}")
            state["cases"].append(result)
            print(f"{result['status']} B={case['batch']} D={case['width']} layout={case['layout']} gap={case['gap']}", flush=True)
        state.update(status="PASS", current_case=None)
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        save(run / "status.json", state)
        args.shared_results.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run, args.shared_results / run.name)
        print("SHARED_ARTIFACTS=" + str(args.shared_results / run.name), flush=True)
    print("M04_ORIGINAL_PASS: 48 GPU cases + 4 empty adapter cases; all values and backing storage checked.")
    print("LIMIT: FP16, int32 unique valid indices, width64/128, two layouts, selected strides. No timing. Triton not tested.")


if __name__ == "__main__":
    main()
