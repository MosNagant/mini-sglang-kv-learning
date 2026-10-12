"""Focused CPU controls for31's new full-matrix binding; no GPU emulation."""
import ast
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("full_grouped_under_test",HERE / "31-check-kv-grouped-full.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def main(output):
    if output.exists():
        raise FileExistsError("REPORT_EXISTS: choose a new path")
    plan = t.make_plan()
    configs = t.c.configurations()
    assert len(configs) == 98 and sum(c["batch"] > 0 for c in configs) == 84
    assert len(t.w.INVALID_KINDS) == 18 and plan["implementation_cases"] == 196
    assert t.run_invalid_cases.__defaults__[0] is t.w.invalid_case
    rejected = []
    class Tensor:
        def __init__(self,shape,stride): self.shape,self.row_stride = shape,stride
        def stride(self,axis):
            assert axis == 0
            return self.row_stride
    class Kernel:
        def __init__(self): self.calls = []
        def __getitem__(self,grid):
            def call(*args,**kwargs): self.calls.append(dict(grid=list(grid),parameters=kwargs))
            return call
    kernels = {arm:Kernel() for arm in t.ARMS}
    apis = dict(frozen_v1=SimpleNamespace(store_kv_v1=None),
                grouped=SimpleNamespace(store_kv_grouped=kernels["grouped"]))
    probes = t.bind_probes(kernels["frozen_v1"],apis)
    assert apis["frozen_v1"].store_kv_v1 is probes["frozen_v1"]
    assert apis["grouped"].store_kv_grouped is probes["grouped"]
    forwarded,empty_blocked,oracle_positives = 0,0,0
    for cfg in configs:
        before = t.c.fixture(cfg)
        t.h.verify_backing(before,t.c.correct_after(before))
        oracle_positives += 1
        b,d = cfg["batch"],cfg["width"]
        src = d+cfg["gap"]
        dst = (d+cfg["guard"])*(2 if cfg["layout"] == "interleaved" else 1)
        args = [Tensor((32,d),dst),Tensor((32,d),dst),Tensor((b,),1),Tensor((b,d),src),Tensor((b,d),src)]
        for arm in t.ARMS:
            probe = probes[arm]
            probe.reset(b > 0)
            params = t.m.launch_parameters(b,d,src,dst,arm)
            grid = ((b+3)//4 if arm == "grouped" else b,)
            prior_calls = len(kernels[arm].calls)
            if b:
                probe[grid](*args,**params)
                assert kernels[arm].calls[-1] == probe.attempts[-1]
                forwarded += 1
            else:
                try:
                    probe[grid](*args,**params)
                except RuntimeError:
                    empty_blocked += 1
                else:
                    raise AssertionError("B0_LAUNCH_NOT_BLOCKED")
                assert len(kernels[arm].calls) == prior_calls
    assert forwarded == 168 and empty_blocked == 28
    accepted = HERE / "logs/m07-grouped-results" / t.m.ACCEPTED_RUN
    gate = t.m.accepted_gate(accepted)
    with tempfile.TemporaryDirectory(prefix="m07-grouped-full-cpu-") as tmp:
        root = Path(tmp)
        sources = root / "sources"
        sources.mkdir()
        for name in (*t.q.PINNED,t.r.GROUPED):
            shutil.copyfile(accepted / name,sources / name)
        for name in t.PINNED_30:
            shutil.copyfile(HERE / name,sources / name)
        t.verify_bound_sources(sources)
        for name in (t.r.GROUPED,"30-compare-kv-grouped.py"):
            old = (sources/name).read_bytes()
            try:
                (sources/name).write_text("changed",encoding="utf-8")
                try:
                    t.verify_bound_sources(sources)
                except RuntimeError:
                    rejected.append("source_change_"+name)
                else:
                    raise AssertionError("SOURCE_CHANGE_NOT_REJECTED")
            finally:
                (sources/name).write_bytes(old)
        run = root / "invalid-dispatch"
        run.mkdir()
        seen = []
        def invalid_stub(torch,api,probe,kind,directory):
            arm = next(arm for arm in t.ARMS if apis[arm] is api)
            assert probe is probes[arm] and directory.name == f"reject-{arm}-{kind}"
            probe.reset(False)
            seen.append((arm,kind))
            return dict(status="REJECTED_BEFORE_KV_LAUNCH",kind=kind,rejection=dict(type="ValueError",message="CPU dispatch stub"))
        state = dict(status="RUNNING",rejected=[])
        t.run_invalid_cases(None,apis,probes,run,state,invalid_runner=invalid_stub)
        assert seen == [(arm,kind) for arm in t.ARMS for kind in t.w.INVALID_KINDS]
        assert len(state["rejected"]) == 36 and all(x["submissions"] == 0 for x in state["rejected"])
        def bad_launch(*args):
            raise RuntimeError("first invalid launch failure")
        failed = dict(status="RUNNING",rejected=[])
        try:
            t.run_invalid_cases(None,apis,probes,run,failed,invalid_runner=bad_launch)
        except RuntimeError as error:
            t.save_failure(run,failed,error)
            assert str(error) == "first invalid launch failure"
        else:
            raise AssertionError("RUNTIME_ERROR_SWALLOWED_AS_CONTRACT_REJECTION")
        assert failed["status"] == "FAILED" and failed["rejected"] == []
        assert "first invalid launch failure" in (run/"error.txt").read_text(encoding="utf-8")
        rejected.append("unexpected_invalid_RuntimeError_preserved")
    for name in (*t.NEW_FILES,"31-grouped-full-cpu-test.py"):
        raw = (HERE/name).read_bytes()
        assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
        if name.endswith(".py"):
            ast.parse(raw.decode("utf-8"))
    assert "torch" not in sys.modules and "triton" not in sys.modules
    report = dict(status="CPU_FULL_GROUPED_BINDING_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",plan=plan,
        source_sha256={name:t.q.sha(HERE/name) for name in (*t.NEW_FILES,"31-grouped-full-cpu-test.py")},
        reused_fixture_oracle_positives=oracle_positives,mock_nonempty_forwards=forwarded,mock_empty_launches_blocked=empty_blocked,
        invalid_dispatches=36,source_and_failure_controls=rejected,accepted_gate=gate,
        limitation="CPU probes/oracles and stub dispatcher only; invalid contract behavior and kernel execution still require user GPU run")
    output.parent.mkdir(parents=True,exist_ok=True)
    t.h.write_json(output,report)
    print(f"CPU_FULL_GROUPED_BINDING_CONTROLS_PASS GPU_NOT_RUN REPORT={output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B 31-grouped-full-cpu-test.py NEW_REPORT_PATH")
    main(Path(sys.argv[1]))
