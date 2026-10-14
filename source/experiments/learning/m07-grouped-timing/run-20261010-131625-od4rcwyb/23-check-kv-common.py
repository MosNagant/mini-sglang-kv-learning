"""M06-05: common original-CUDA / learner-public-Triton correctness matrix.

Coach runner reuses M04 fixtures/oracle and M06 mask/wrapper source checks.
98 configurations x 2 implementations: 168 nonempty submissions + 28 empty
adapter returns. No timing. Never rewrites learner files or accepted evidence.
"""
import argparse
import hashlib
import importlib.util
import inspect
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

w = helper("common_wrapper_helpers", "22-check-kv-wrapper.py")
base = helper("common_m04_helpers", "15-check-kv-contract.py")
h, m, COMMIT = w.h, w.m, w.COMMIT
BATCHES = (0, 1, 2, 3, 4, 5, 17)
IMPLEMENTATIONS = ("original_cuda", "learner_triton")


def configurations():
    result = []
    for layout_cfg in m.configurations():
        for batch in BATCHES:
            rows = ([31, 0] + list(range(1, batch - 1)))[:batch]
            result.append(dict(batch=batch, width=layout_cfg["width"], capacity=32,
                               layout=layout_cfg["layout"], gap=layout_cfg["input_gap"],
                               guard=layout_cfg["cache_guard"], rows=rows))
    return result


def fixture(cfg):
    if cfg not in configurations():
        raise ValueError("COMMON_CONFIGURATION_OUT_OF_SCOPE")
    if cfg["width"] in (64, 128):
        # The original M04 finite-value fixtures are preserved exactly.
        data = base.fixture(cfg)
        data["expected_cache"] = data.pop("expected")
        return data
    b, d, c = (cfg[k] for k in ("batch", "width", "capacity"))
    pitch = d + cfg["guard"]
    shape = (c, 2, pitch) if cfg["layout"] == "interleaved" else (2, c, pitch)
    cache = np.full(shape, -1, dtype=np.float16)
    kc, vc = h.cache_views(cache, cfg["layout"])
    kc[:, d:] = -7
    vc[:, d:] = -7
    data = dict(cache=cache, k=np.full((b, d + cfg["gap"]), -9, dtype=np.float16),
                v=np.full((b, d + cfg["gap"]), -9, dtype=np.float16),
                indices=np.array(cfg["rows"], dtype=np.int32))
    # Bounded and exactly representable, distinct K/V ranges; no FP16 overflow.
    values = (np.arange(b*d).reshape(b, d) % 97).astype(np.float16)
    data["k"][:, :d] = values + 10
    data["v"][:, :d] = values + 1000
    data["expected_cache"] = cache.copy()
    ek, ev = h.cache_views(data["expected_cache"], cfg["layout"])
    for pid, pos in enumerate(cfg["rows"]):
        ek[pos, :d] = data["k"][pid, :d]
        ev[pos, :d] = data["v"][pid, :d]
    return data


def correct_after(before):
    return {n: before["expected_cache" if n == "cache" else n].copy()
            for n in ("cache", "k", "v", "indices")}


def full_storage_controls():
    records = []
    for cfg in configurations():
        before = fixture(cfg)
        h.verify_backing(before, correct_after(before))
        b, d = cfg["batch"], cfg["width"]
        kinds = ["untouched_K", "untouched_V"]
        if b:
            kinds += ["target_K_last", "target_V_last", "input_K", "input_V", "indices"]
        if b >= 2:
            kinds += ["wrong_mapping", "swap_KV"]
        if b and cfg["gap"]:
            kinds += ["input_K_gap", "input_V_gap"]
        if cfg["guard"]:
            kinds += ["cache_K_guard_first", "cache_K_guard_last",
                      "cache_V_guard_first", "cache_V_guard_last"]
        caught = []
        for kind in kinds:
            after = correct_after(before)
            ak, av = h.cache_views(after["cache"], cfg["layout"])
            unused = next(i for i in range(cfg["capacity"]) if i not in cfg["rows"])
            if kind == "untouched_K": ak[unused, 0] = -99
            elif kind == "untouched_V": av[unused, 0] = -99
            elif kind == "target_K_last": ak[cfg["rows"][-1], d-1] = -99
            elif kind == "target_V_last": av[cfg["rows"][-1], d-1] = -99
            elif kind == "input_K": after["k"][0, 0] = -99
            elif kind == "input_V": after["v"][0, 0] = -99
            elif kind == "indices": after["indices"][0] = unused
            elif kind == "input_K_gap": after["k"][0, d] = -99
            elif kind == "input_V_gap": after["v"][0, d] = -99
            elif kind == "cache_K_guard_first": ak[0, d] = -99
            elif kind == "cache_K_guard_last": ak[-1, -1] = -99
            elif kind == "cache_V_guard_first": av[0, d] = -99
            elif kind == "cache_V_guard_last": av[-1, -1] = -99
            elif kind == "wrong_mapping":
                p, q = cfg["rows"][:2]
                ak[p, :d], ak[q, :d] = ak[q, :d].copy(), ak[p, :d].copy()
            elif kind == "swap_KV":
                p = cfg["rows"][0]
                ak[p, :d], av[p, :d] = av[p, :d].copy(), ak[p, :d].copy()
            try:
                h.verify_backing(before, after)
            except AssertionError:
                caught.append(kind)
            else:
                raise AssertionError("COMMON_CHECKER_MISSED: " + kind)
        records.append(dict(config=cfg, positive="PASS", corruptions_rejected=caught))
    return records


def cpu_self_test():
    configs = configurations()
    assert len(configs) == 98
    assert sum(c["batch"] > 0 for c in configs) == 84
    assert sum(c["batch"] == 0 for c in configs) == 14
    # Include all 48+4 original M04 cases, not merely similar configurations.
    for old in base.cases():
        matching = [c for c in configs if c == dict(old, guard=0)]
        assert len(matching) == 1
        old_data, new_data = base.fixture(old), fixture(matching[0])
        for n, value in old_data.items():
            np.testing.assert_array_equal(value, new_data["expected_cache" if n == "expected" else n])
    controls = full_storage_controls()
    for cfg in configs:
        data = fixture(cfg)
        assert all(np.isfinite(a).all() for a in data.values())
    return dict(status="CPU_FIXTURE_CONTROLS_PASS", GPU="NOT_RUN",
                configurations=98, original_M04_fixtures_unchanged=52,
                backing_controls=controls, m04_controls=base.controls(),
                limitation="No PyTorch/Triton compilation or GPU result inferred")


def learner_checker_controls(torch, checker):
    """Exercise the learner's function by itself, including last input B17."""
    records = []
    for b in (1, 3, 17):
        for d in (64, 128, 192):
            cfg = next(c for c in configurations() if c["batch"] == b and c["width"] == d
                       and c["layout"] == "interleaved" and c["gap"] == c["guard"] == 0)
            data = fixture(cfg)
            ck, cv = h.cache_views(data["expected_cache"], cfg["layout"])
            def call(ak, av):
                checker(*(torch.from_numpy(a.copy()) for a in
                          (ak[:, :d], av[:, :d], data["k"][:, :d], data["v"][:, :d])), cfg["rows"])
            call(ck, cv)
            caught = []
            for name in ("K_last_input_last_column", "V_last_input_last_column"):
                ak, av = ck.copy(), cv.copy()
                (ak if name.startswith("K") else av)[cfg["rows"][-1], d-1] = -99
                try:
                    call(ak, av)
                except AssertionError:
                    caught.append(name)
                else:
                    raise AssertionError("LEARNER_CHECK_MISSED: " + name)
            records.append(dict(batch=b, width=d, positive="PASS", corruptions_rejected=caught))
    return records


def gpu_packet(torch, cfg, before):
    buffers = {n: torch.from_numpy(before[n].copy()).cuda()
               for n in ("cache", "k", "v", "indices")}
    d = cfg["width"]
    if cfg["batch"] == 0:
        for n in ("k", "v"):
            buffers[n] = torch.empty((0, d + cfg["gap"]), dtype=torch.float16, device="cuda")
        buffers["indices"] = torch.empty((0,), dtype=torch.int32, device="cuda")
    kc, vc = h.cache_views(buffers["cache"], cfg["layout"])
    args = [kc[:, :d], vc[:, :d], buffers["indices"],
            buffers["k"][:, :d], buffers["v"][:, :d]]
    return buffers, args


class OriginalAdapter:
    """Coach adapter for common validation and B0; no upstream B0 support claim."""
    def __init__(self, original, support):
        self.original, self.support = original, support
        self.attempts = []

    def __call__(self, kc, vc, indices, k, v):
        b, d, src, dst, block = self.support.validate_metadata(kc, vc, indices, k, v)
        if indices.shape != (b,):
            raise ValueError("INDEX_COUNT")
        if b == 0:
            return
        self.support.validate_index_values(indices, b, kc.shape[0])
        self.attempts.append(dict(input_rows=b, width=d, src_stride=src, dst_stride=dst,
                                  meaning="one original public operator call; not a measured kernel count"))
        return self.original(kc, vc, indices, k, v)


def run_case(torch, original_adapter, api, probe, checker, cfg, implementation, directory, before):
    directory.mkdir()
    # Fresh allocation per implementation; neither inherits the other's output.
    buffers, args = gpu_packet(torch, cfg, before)
    h.write_json(directory / "metadata.json", dict(config=cfg, implementation=implementation,
                 tensors=[w.tensor_info(t) for t in args]))
    np.savez_compressed(directory / "before.npz", **before)
    probe.reset(cfg["batch"] > 0)
    original_adapter.attempts = []
    attempts = probe.attempts if implementation == "learner_triton" else original_adapter.attempts
    record = dict(status="RUNNING", implementation=implementation, config=cfg)
    h.write_json(directory / "status.json", record)
    try:
        if implementation == "learner_triton":
            api.store_cache_triton(*args)
        else:
            original_adapter(*args)
        torch.cuda.synchronize()
        after = {n: t.cpu().numpy().copy() for n, t in buffers.items()}
        np.savez_compressed(directory / "after.npz", **after)
        h.verify_backing(before, after)
        ck, cv = h.cache_views(after["cache"], cfg["layout"])
        d = cfg["width"]
        checker(torch.from_numpy(ck[:, :d].copy()), torch.from_numpy(cv[:, :d].copy()),
                torch.from_numpy(before["k"][:, :d].copy()),
                torch.from_numpy(before["v"][:, :d].copy()), cfg["rows"])
        if len(attempts) != int(cfg["batch"] > 0):
            raise AssertionError("SUBMISSION_COUNT_MISMATCH")
        record.update(status="PASS" if cfg["batch"] else "EMPTY_ADAPTER_PASS",
                      submissions=len(attempts))
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        h.write_json(directory / "submissions.json", attempts)
        h.write_json(directory / "status.json", record)
    print(f"COMMON_CASE_PASS implementation={implementation} B={cfg['batch']} D={cfg['width']} "
          f"layout={cfg['layout']} input_gap={cfg['gap']} cache_guard={cfg['guard']} "
          f"submissions={record['submissions']}", flush=True)
    return record, after


def snapshot(project, run, git):
    filenames = ["23-check-kv-common.py", "23-check-kv-common.sh", "22-check-kv-wrapper.py",
                 "22-triton-store-api-template.py", "22-kv-contract-support.py",
                 "21-check-kv-mask.py", "21-triton-store-mask.py", "14-check-kv-widths.py",
                 "15-check-kv-contract.py"]
    for name in filenames:
        shutil.copyfile(HERE / name, run / name)
    for name in ("triton_store_api.py", "kv_contract_support.py", "triton_store_mask.py", "check_kv_targets.py"):
        shutil.copyfile(project / "experiments" / name, run / name)
    for relative in git("ls-files", "--", "python/minisgl/kernel").splitlines():
        destination = run / "original_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, destination)
    hashes = {str(p.relative_to(run)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in run.rglob("*") if p.is_file()}
    h.write_json(run / "source_hashes.json", hashes)
    if hashes["kv_contract_support.py"] != hashes["22-kv-contract-support.py"]:
        raise RuntimeError("SUPPORT_CHANGED: retained; review before launch")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--cpu-self-test", type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED: run without -O")
    if args.cpu_self_test:
        result = cpu_self_test()
        args.cpu_self_test.parent.mkdir(parents=True, exist_ok=True)
        h.write_json(args.cpu_self_test, result)
        print("CPU_FIXTURE_CONTROLS_PASS: 98 configurations; GPU NOT_RUN")
        return
    if args.project is None or args.shared_results is None:
        parser.error("--project and --shared-results required")
    project = args.project.resolve()
    def git(*words):
        return subprocess.check_output(["git", "-C", str(project), *words], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain", "--", "python/minisgl/kernel"):
        raise RuntimeError("PINNED_ORIGINAL_CHANGED: no files rewritten")
    for name in ("triton_store_api.py", "kv_contract_support.py", "triton_store_mask.py", "check_kv_targets.py"):
        if not (project / "experiments" / name).is_file():
            raise FileNotFoundError("Accepted lesson source required: " + name)
    parent = project / "experiments/learning/m06-common"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = dict(status="PREPARING", lesson="M06-05", commit=COMMIT, cases=[], pairs=[],
                 expected_implementation_cases=196, expected_nonempty_calls=168, expected_empty_returns=28,
                 configurations=configurations(), scope="common correctness; FP16/int32; C32; no timing",
                 ownership="Learner's reviewed kernel/wrapper/checks reused; matrix, oracle and records by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['scope']}", flush=True)
    current_dir = None
    try:
        snapshot(project, run, git)
        source_review = dict(wrapper=w.review_api((run/"triton_store_api.py").read_text(encoding="utf-8-sig")),
                             mask=m.validate_mask_source((run/"triton_store_mask.py").read_text(encoding="utf-8-sig")))
        h.write_json(run / "source_review.json", source_review)
        h.write_json(run / "fixture_controls.json", cpu_self_test())
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if Path(inspect.getsourcefile(store_cache)).resolve() != (project/"python/minisgl/kernel/store.py").resolve():
            raise RuntimeError("WRONG_MINISGL_IMPORT")
        support = h.load_module(run / "kv_contract_support.py", "kv_contract_support")
        kernel = h.load_module(run / "triton_store_mask.py", "triton_store_mask")
        api = h.load_module(run / "triton_store_api.py", "common_learner_api")
        checker = h.load_module(run / "check_kv_targets.py", "common_learner_checker").check_kv_targets
        h.write_json(run / "checker_controls.json", learner_checker_controls(torch, checker))
        print("CONTROLS_PASS: independent full-storage oracle and learner K/V corruption checks; no invalid KV launches", flush=True)
        probe = w.LaunchProbe(kernel.store_kv_v1)
        api.store_kv_v1 = probe
        original_adapter = OriginalAdapter(store_cache, support)
        state.update(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0), public_api_policy="index values checked on CPU; may synchronize; no timing",
                     empty_policy="original coach adapter and learner public wrapper return without KV submission")
        for index, cfg in enumerate(configurations()):
            # Build one independent CPU oracle, then allocate each GPU case anew.
            before = fixture(cfg)
            results = []
            for implementation in IMPLEMENTATIONS:
                current_dir = run / f"case-{index:03d}-{implementation}"
                state.update(status="RUNNING", current_case=current_dir.name)
                h.write_json(run / "status.json", state)
                record, actual = run_case(torch, original_adapter, api, probe, checker, cfg,
                                          implementation, current_dir, before)
                state["cases"].append(record)
                results.append(actual)
            for name in ("cache", "k", "v", "indices"):
                np.testing.assert_array_equal(results[0][name], results[1][name], err_msg="PAIR_"+name)
            state["pairs"].append(dict(config=cfg, status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"))
        assert len(state["cases"]) == 196 and len(state["pairs"]) == 98
        state.update(status="PASS", current_case=None)
    except Exception as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json", state)
        args.shared_results.mkdir(parents=True, exist_ok=True)
        shared = args.shared_results / run.name
        try:
            shutil.copytree(run, shared)
            print(f"SHARED_ARTIFACTS={shared}", flush=True)
        except OSError as error:
            print(f"SHARED_COPY_FAILED={error}; WSL evidence retained at {run}", flush=True)
            if state["status"] == "PASS": raise
    print("COMMON_CORRECTNESS_PASS: 196/196 = 168 nonempty calls + 28 empty adapter returns; 98 pairs match independent reference")
    print("COVERAGE_LIMIT: B0/1/2/3/4/5/17 C32 D64/128/192; selected layouts/gaps/guards; FP16/int32 unique valid indices; no timing")
    print("REVIEW_REQUIRED: saved source/results and learner explanation; no automatic M06 or performance acceptance")


if __name__ == "__main__":
    main()
