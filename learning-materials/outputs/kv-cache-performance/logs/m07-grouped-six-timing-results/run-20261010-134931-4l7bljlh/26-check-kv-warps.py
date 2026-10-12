"""M07-01/02 coach scaffolding: frozen v1=4 versus learner candidate=1.

--prepare creates the candidate launcher once, with NUM_WARPS still 4.
Default: 98 common configurations per arm, 18 invalid public calls per arm.
No original CUDA launch, timing, automatic score or performance claim.
"""
import argparse
import ast
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("warps_common_helpers", HERE / "23-check-kv-common.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
h, w, m, COMMIT = c.h, c.w, c.m, c.COMMIT
ARMS = {"frozen_v1": 4, "candidate": 1}
CANDIDATE = "triton_store_candidate.py"
# Byte hashes from the accepted, archived M06 sources. No live file is rewritten.
PINNED = {
    "triton_store_api.py": "05eff6e46b214fc19e4a11152030665916e798cfdc17bd8cf0725f79f84cc35f",
    "triton_store_mask.py": "6e76ef9eb319f4e57e4a55923fcb5655fa5fb26f29c8c7af00c49f0c125bb19d",
    "kv_contract_support.py": "0566a788f2a4984952815d99c0c7aa82e3af43829d2c71630c1be707f582a881",
    "check_kv_targets.py": "df82392fe0463edf8084ded9dc0be576b93f0e4c00a96dca87d492495a434358",
}
HELPERS = (
    "26-check-kv-warps.py", "26-check-kv-warps.sh", "26-triton-store-candidate.py",
    "23-check-kv-common.py", "22-check-kv-wrapper.py", "22-triton-store-api-template.py",
    "22-kv-contract-support.py", "21-check-kv-mask.py", "21-triton-store-mask.py",
    "14-check-kv-widths.py", "15-check-kv-contract.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pinned(directory):
    for name, expected in PINNED.items():
        if sha(directory / name) != expected:
            raise RuntimeError("ACCEPTED_SOURCE_CHANGED: " + name + "; preserve and review; no overwrite")
    return dict(PINNED)


def candidate_review(source, require_change=True):
    tree = ast.parse(source)
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == "NUM_WARPS"]
    if len(constants) != 1 or not isinstance(constants[0].value, ast.Constant):
        raise RuntimeError("CANDIDATE_SOURCE_REVIEW_REQUIRED: expected NUM_WARPS literal")
    value = constants[0].value.value
    if type(value) is not int or value not in (1, 4):
        raise RuntimeError("CANDIDATE_SOURCE_REVIEW_REQUIRED: first experiment is 4 to 1")
    constants[0].value = ast.Constant(value=4)
    expected = ast.parse((HERE / "26-triton-store-candidate.py").read_text(encoding="utf-8"))
    if ast.dump(tree) != ast.dump(expected):
        raise RuntimeError("CANDIDATE_SOURCE_REVIEW_REQUIRED: only NUM_WARPS may change")
    if require_change and value != 1:
        raise RuntimeError("LEARNER_CHANGE_PENDING: change NUM_WARPS = 4 to NUM_WARPS = 1 in " + CANDIDATE)
    return dict(status="ONE_PARAMETER_REVIEWED", num_warps=value,
                kernel="accepted triton_store_mask.py reused unchanged", grid="(batch,)",
                ownership="scaffold by coach; parameter change by learner; independent mastery not inferred")


def prepare(project):
    exp = project / "experiments"
    check_pinned(exp)
    w.install_once(HERE / "26-triton-store-candidate.py", exp / CANDIDATE)
    candidate_review((exp / CANDIDATE).read_text(encoding="utf-8-sig"), require_change=False)
    return exp / CANDIDATE


class LaunchProbe:
    """Record and validate actual launch arguments before forwarding the call."""
    def __init__(self, kernel, expected_warps):
        self.kernel, self.expected_warps = kernel, expected_warps
        self.reset(False)

    def reset(self, allow):
        self.allow, self.attempts = allow, []

    def __getitem__(self, grid):
        def call(kc, vc, indices, k, v, **kwargs):
            self.attempts.append(dict(grid=list(grid), parameters=kwargs))
            b, d = k.shape
            expected = dict(WIDTH=d, SRC_ROW_STRIDE=k.stride(0), DST_ROW_STRIDE=kc.stride(0),
                            BLOCK_SIZE=1 << (d-1).bit_length(), num_warps=self.expected_warps)
            if not self.allow or grid != (b,) or kwargs != expected or b == 0 or indices.shape != (b,):
                raise RuntimeError("BAD_LAUNCH_BLOCKED_BY_TEST")
            return self.kernel[grid](kc, vc, indices, k, v, **kwargs)
        return call


def run_case(torch, api, probe, checker, cfg, arm, directory, before):
    directory.mkdir()
    buffers, args = c.gpu_packet(torch, cfg, before)
    h.write_json(directory / "metadata.json", dict(config=cfg, implementation=arm,
                 tensors=[w.tensor_info(t) for t in args]))
    np.savez_compressed(directory / "before.npz", **before)
    record = dict(status="RUNNING", implementation=arm, config=cfg)
    h.write_json(directory / "status.json", record)
    probe.reset(cfg["batch"] > 0)
    try:
        api.store_cache_triton(*args)
        torch.cuda.synchronize()
        after = {n: t.cpu().numpy().copy() for n, t in buffers.items()}
        np.savez_compressed(directory / "after.npz", **after)
        h.verify_backing(before, after)
        ck, cv = h.cache_views(after["cache"], cfg["layout"])
        d = cfg["width"]
        checker(*(torch.from_numpy(a.copy()) for a in
                  (ck[:, :d], cv[:, :d], before["k"][:, :d], before["v"][:, :d])), cfg["rows"])
        if len(probe.attempts) != int(cfg["batch"] > 0):
            raise AssertionError("SUBMISSION_COUNT_MISMATCH")
        record.update(status="PASS" if cfg["batch"] else "EMPTY_WRAPPER_PASS", submissions=len(probe.attempts))
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        h.write_json(directory / "submissions.json", probe.attempts)
        h.write_json(directory / "status.json", record)
    print(f"WARPS_CASE_PASS arm={arm} B={cfg['batch']} D={cfg['width']} "
          f"layout={cfg['layout']} gap={cfg['gap']} guard={cfg['guard']} submissions={record['submissions']}", flush=True)
    return record, after


def snapshot(project, run):
    for name in HELPERS:
        shutil.copyfile(HERE / name, run / name)
    for name in (*PINNED, CANDIDATE):
        shutil.copyfile(project / "experiments" / name, run / name)
    hashes = {p.name: sha(p) for p in run.iterdir() if p.is_file()}
    h.write_json(run / "source_hashes.json", hashes)
    check_pinned(run)
    h.write_json(run / "source_review.json", dict(
        frozen_sources=PINNED,
        v1=w.review_api((run / "triton_store_api.py").read_text(encoding="utf-8-sig")),
        kernel=m.validate_mask_source((run / "triton_store_mask.py").read_text(encoding="utf-8-sig")),
        candidate=candidate_review((run / CANDIDATE).read_text(encoding="utf-8-sig"))))


def cpu_self_test():
    common = c.cpu_self_test()
    template = (HERE / "26-triton-store-candidate.py").read_text(encoding="utf-8")
    changed = template.replace("NUM_WARPS = 4", "NUM_WARPS = 1")
    assert candidate_review(changed)["num_warps"] == 1
    rejected = []
    for name, source in (
        ("unchanged_4", template),
        ("warps_2", template.replace("NUM_WARPS = 4", "NUM_WARPS = 2")),
        ("grid_changed", changed.replace("[(batch,)]", "[(1,)]")),
        ("block_changed", changed.replace("BLOCK_SIZE=block", "BLOCK_SIZE=32")),
        ("index_check_removed", changed.replace("indices.shape != (batch,)", "False")),
    ):
        try:
            candidate_review(source)
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError("MISSED_SOURCE_CHANGE: " + name)
    with tempfile.TemporaryDirectory(prefix="m07-cpu-prepare-") as tmp:
        project = Path(tmp)
        exp = project / "experiments"
        exp.mkdir()
        for name in PINNED:
            shutil.copyfile(HERE / "modules/M06/src" / name, exp / name)
        path = prepare(project)
        assert path.read_text(encoding="utf-8") == template
        path.write_text(changed, encoding="utf-8", newline="\n")
        saved = path.read_bytes()
        assert prepare(project).read_bytes() == saved
        check_pinned(exp)
        (exp / "triton_store_mask.py").write_text("changed", encoding="utf-8")
        try:
            prepare(project)
        except RuntimeError as error:
            assert "ACCEPTED_SOURCE_CHANGED" in str(error)
        else:
            raise AssertionError("CHANGED_PIN_ACCEPTED")
        assert path.read_bytes() == saved
    class Tensor:
        def __init__(self, shape, stride): self.shape, self.row_stride = shape, stride
        def stride(self, dimension):
            assert dimension == 0
            return self.row_stride
    class Kernel:
        def __init__(self): self.calls = []
        def __getitem__(self, grid):
            def call(*args, **kwargs): self.calls.append(dict(grid=list(grid), parameters=kwargs))
            return call
    forwards, blocked = 0, 0
    for cfg in c.configurations():
        b, d = cfg["batch"], cfg["width"]
        src = d + cfg["gap"]
        dst = (d + cfg["guard"]) * (2 if cfg["layout"] == "interleaved" else 1)
        args = [Tensor((32, d), dst), Tensor((32, d), dst), Tensor((b,), 1),
                Tensor((b, d), src), Tensor((b, d), src)]
        for arm, warps in ARMS.items():
            kernel = Kernel()
            probe = LaunchProbe(kernel, warps)
            probe.reset(b > 0)
            kwargs = dict(WIDTH=d, SRC_ROW_STRIDE=src, DST_ROW_STRIDE=dst,
                          BLOCK_SIZE=1 << (d-1).bit_length(), num_warps=warps)
            if b:
                probe[(b,)](*args, **kwargs)
                assert kernel.calls == probe.attempts
                forwards += 1
            for grid, bad in (((b,), dict(kwargs, num_warps=5-warps)), ((b+1,), kwargs)):
                before = len(kernel.calls)
                try:
                    probe[grid](*args, **bad)
                except RuntimeError:
                    blocked += 1
                else:
                    raise AssertionError("BAD_PROBE_FORWARD")
                assert len(kernel.calls) == before
    return dict(status="CPU_HARNESS_CONTROLS_PASS", GPU="NOT_RUN", timing="NOT_IMPLEMENTED",
                common_fixture_controls=common, source_mutations_rejected=rejected,
                prepare="creates once; preserves learner edit; rejects changed accepted kernel",
                mock_nonempty_forwards=forwards, mock_bad_launches_blocked=blocked,
                limitation="CPU fixtures, AST, file installation and launch-probe mocks only; no GPU correctness inferred")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cpu-self-test", type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED: run without -O")
    if args.cpu_self_test:
        if args.cpu_self_test.exists():
            raise FileExistsError("CPU_REPORT_EXISTS: choose a new report path")
        result = cpu_self_test()
        args.cpu_self_test.parent.mkdir(parents=True, exist_ok=True)
        h.write_json(args.cpu_self_test, result)
        print("CPU_HARNESS_CONTROLS_PASS: GPU NOT_RUN; no timing")
        return
    if args.project is None or (not args.prepare and args.shared_results is None):
        parser.error("--project required; default checks also require --shared-results")
    project = args.project.resolve()
    if args.prepare:
        path = prepare(project)
        print(f"CANDIDATE_PATH={path}\nPREPARE_COMPLETE: candidate retained; learner changes NUM_WARPS 4 to 1; GPU NOT_RUN")
        return
    parent = project / "experiments/learning/m07-warps"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = dict(status="PREPARING", lesson="M07-01/02", commit=COMMIT, arms=ARMS,
                 cases=[], pairs=[], rejected=[], configurations=c.configurations(),
                 expected_cases=196, expected_nonempty_calls=168, expected_empty_returns=28,
                 expected_invalid_rejections=36,
                 scope="FP16/int32 C32 B0/1/2/3/4/5/17 D64/128/192; selected layouts/gaps/guards",
                 public_api_policy="index values checked; may synchronize; correctness only",
                 timing="NOT_IMPLEMENTED: future trusted launcher must be separate from public validation",
                 ownership="coach runner; candidate parameter edited by learner; no mastery inferred")
    print(f"RUN_DIRECTORY={run}", flush=True)
    try:
        snapshot(project, run)
        commit = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
        if commit != COMMIT:
            raise RuntimeError("PINNED_PROJECT_COMMIT_CHANGED")
        h.write_json(run / "fixture_controls.json", c.cpu_self_test())
        import torch
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if (torch.__version__, torch.version.cuda, triton.__version__) != ("2.9.1+cu128", "12.8", "3.5.1"):
            raise RuntimeError("FIXED_ENVIRONMENT_CHANGED: review without automatic upgrade")
        h.load_module(run / "kv_contract_support.py", "kv_contract_support")
        kernel = h.load_module(run / "triton_store_mask.py", "triton_store_mask")
        apis = {"frozen_v1": h.load_module(run / "triton_store_api.py", "warps_v1_api"),
                "candidate": h.load_module(run / CANDIDATE, "warps_candidate_api")}
        checker = h.load_module(run / "check_kv_targets.py", "warps_checker").check_kv_targets
        h.write_json(run / "checker_controls.json", c.learner_checker_controls(torch, checker))
        probes = {arm: LaunchProbe(kernel.store_kv_v1, warps) for arm, warps in ARMS.items()}
        for arm in ARMS:
            apis[arm].store_kv_v1 = probes[arm]
        state.update(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0))
        for index, cfg in enumerate(c.configurations()):
            before, results = c.fixture(cfg), []
            for arm in ARMS:
                directory = run / f"case-{index:03d}-{arm}"
                state.update(status="RUNNING", current_case=directory.name)
                h.write_json(run / "status.json", state)
                record, actual = run_case(torch, apis[arm], probes[arm], checker, cfg, arm, directory, before)
                state["cases"].append(record)
                results.append(actual)
            for name in ("cache", "k", "v", "indices"):
                np.testing.assert_array_equal(results[0][name], results[1][name], err_msg="PAIR_" + name)
            state["pairs"].append(dict(config=cfg, status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"))
        for arm in ARMS:
            for kind in w.INVALID_KINDS:
                directory = run / f"reject-{arm}-{kind}"
                state.update(current_case=directory.name)
                h.write_json(run / "status.json", state)
                record = w.invalid_case(torch, apis[arm], probes[arm], kind, directory)
                state["rejected"].append(dict(record, implementation=arm))
        assert len(state["cases"]) == 196 and len(state["pairs"]) == 98 and len(state["rejected"]) == 36
        state.update(status="PASS", current_case=None)
    except Exception as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json", state)
        try:
            args.shared_results.mkdir(parents=True, exist_ok=True)
            shared = args.shared_results / run.name
            shutil.copytree(run, shared)
            print(f"SHARED_ARTIFACTS={shared}", flush=True)
        except OSError as error:
            print(f"SHARED_COPY_FAILED={error}; WSL evidence retained at {run}", flush=True)
            if state["status"] == "PASS": raise
    print("WARPS_CORRECTNESS_PASS: 196/196 = 168 nonempty calls + 28 empty wrapper returns; 98 pairs; 36 invalid rejections")
    print("LIMIT: common M06 matrix only; B256 timing inputs still require later correctness checks; no timing/speedup/AMD claim")


if __name__ == "__main__":
    main()
