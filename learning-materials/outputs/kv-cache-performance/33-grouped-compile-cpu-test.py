"""CPU-only controls for33; mock launches and re-export of old compiler bytes."""
import ast
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grouped_compile_under_test",HERE / "33-inspect-kv-grouped-compile.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def main(output):
    if output.exists():
        raise FileExistsError("REPORT_EXISTS: choose a new output")
    rejected = []
    def reject(label,call,error=RuntimeError):
        try:
            call()
        except error:
            rejected.append(label)
        else:
            raise AssertionError("NEGATIVE_CONTROL_NOT_REJECTED: " + label)
    class Tensor:
        def __init__(self,shape,stride): self.shape,self.row_stride = shape,stride
        def stride(self,axis):
            assert axis == 0
            return self.row_stride
    class Kernel:
        def __init__(self): self.calls,self.compiled = [],object()
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                self.calls.append(dict(grid=list(grid),parameters=kwargs))
                return self.compiled
            return call
    cases = []
    assert t.make_plan()["checked_calls"] == 4
    for cfg in t.configurations():
        before = t.fixture(cfg)
        good = {n:before["expected_cache" if n == "cache" else n].copy() for n in ("cache","k","v","indices")}
        t.h.verify_backing(before,good)
        d = cfg["width"]
        for label,name,index in (("Ktarget","cache",(255,0,d-1)),("Vtarget","cache",(255,1,d-1)),
            ("untouched","cache",(1023,0,0)),("Kinput","k",(255,d-1)),("Vinput","v",(255,d-1)),("index","indices",255)):
            bad = {n:a.copy() for n,a in good.items()}
            bad[name][index] = -99
            reject(f"D{d}-{label}",lambda:t.h.verify_backing(before,bad),AssertionError)
        bad = {n:a.copy() for n,a in good.items()}
        bad["cache"][[0,8]] = bad["cache"][[8,0]]
        reject(f"D{d}-wrong-row-mapping",lambda:t.h.verify_backing(before,bad),AssertionError)
        for arm in t.ARMS:
            args = [Tensor((1024,d),2*d),Tensor((1024,d),2*d),Tensor((256,),1),Tensor((256,d),d),Tensor((256,d),d)]
            kernel = Kernel()
            probe = t.CompiledCapture(kernel,arm,cfg)
            params = t.t.launch_parameters(256,d,d,2*d,arm)
            grid = (64 if arm == "grouped" else 256,)
            probe.reset(True)
            assert probe[grid](*args,**params) is kernel.compiled and probe.compiled is kernel.compiled
            assert probe.attempts == kernel.calls == [dict(grid=list(grid),parameters=params)]
            cases.append(dict(width=d,arm=arm,actual_mock_launch=kernel.calls[0]))
            mutations = {
                "grid":((1,),args,params),
                "source-stride":(grid,args,dict(params,SRC_ROW_STRIDE=d+1)),
                "destination-stride":(grid,args,dict(params,DST_ROW_STRIDE=d)),
                "warps":(grid,args,dict(params,num_warps=1)),
                "other-arm-parameters":(grid,args,t.t.launch_parameters(256,d,d,2*d,"frozen_v1" if arm == "grouped" else "grouped")),
            }
            bad_args = copy.deepcopy(args)
            bad_args[3].row_stride += 1
            mutations["actual-source-stride"] = (grid,bad_args,params)
            for label,(bad_grid,bad_args,bad_params) in mutations.items():
                probe.reset(True)
                reject(f"D{d}-{arm}-{label}",lambda:probe[bad_grid](*bad_args,**bad_params))
                assert probe.compiled is None and len(kernel.calls) == 1
            probe.reset(False)
            reject(f"D{d}-{arm}-forward-disabled",lambda:probe[grid](*args,**params))
            assert probe.compiled is None and len(kernel.calls) == 1
    reject("unknown-arm",lambda:t.CompiledCapture(Kernel(),"candidate",t.configurations()[0]),ValueError)
    reject("unplanned-config",lambda:t.fixture(dict(t.configurations()[0],batch=16)),ValueError)
    accepted = HERE / "logs/m07-grouped-six-timing-results" / t.TIMING_RUN
    old_hashes = json.loads((accepted / "source_hashes.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="m07-grouped-compile-cpu-") as temporary:
        root = Path(temporary)
        project = root / "project"
        archived = project / "experiments/learning/m07-grouped-six-timing" / t.TIMING_RUN
        archived.mkdir(parents=True)
        for name in (*old_hashes,"source_hashes.json","status.json"):
            destination = archived / name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(accepted/name,destination)
            if name.startswith("original_source/"):
                live = project / name.removeprefix("original_source/")
                live.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(accepted/name,live)
        for directory in accepted.glob("round-*-B*-D*"):
            if directory.is_dir():
                (archived/directory.name).mkdir()
                shutil.copyfile(directory/"result.json",archived/directory.name/"result.json")
        for name in (*t.q.PINNED,t.r.GROUPED):
            shutil.copyfile(accepted/name,project/"experiments"/name)
        gate_hashes,gate = t.accepted_gate(archived,project)
        assert gate_hashes == old_hashes
        run = root / "observation"
        run.mkdir()
        t.snapshot(project,run)
        t.verify_snapshot(run,project)
        def changed_file(label,path,data,call):
            original = path.read_bytes()
            try:
                path.write_bytes(data)
                reject(label,call)
            finally:
                path.write_bytes(original)
        changed_file("frozen-snapshot-grouped-changed",run/t.r.GROUPED,b"changed",lambda:t.verify_snapshot(run,project))
        changed_file("live-grouped-changed",project/"experiments"/t.r.GROUPED,b"changed",lambda:t.verify_snapshot(run,project))
        changed_file("executed32-helper-changed",run/"32-compare-kv-grouped-full.py",b"changed",lambda:t.verify_local_sources(run,old_hashes))
        changed_file("pinned-export-helper-changed",run/"28-compare-kv-warps-compile.py",b"changed",lambda:t.verify_local_sources(run,old_hashes))
        failed_status = json.loads((archived/"status.json").read_text(encoding="utf-8"))
        failed_status["status"] = "FAILED"
        changed_file("accepted32-noncomplete",archived/"status.json",json.dumps(failed_status).encode(),lambda:t.accepted_gate(archived,project))
        worker_record = next(archived.glob("round-*-B*-D*/result.json"))
        changed_file("accepted32-worker-record-changed",worker_record,b"{}",lambda:t.accepted_gate(archived,project))
        # Reuse actual historical bytes only to check the exporter, not as new evidence.
        old = HERE / "logs/m06-compile-results/run-20261009-142759-l12p0a0c"
        old_report = json.loads((old/"compiled.json").read_text(encoding="utf-8"))
        assembly = {kind:(old/item["path"]).read_bytes() for kind,item in old_report["artifacts"].items()}
        destination = root / "old-byte-export"
        destination.mkdir()
        report = t.e.export_compiled(SimpleNamespace(asm=assembly,metadata={},name="CPU_OLD_BYTE_EXPORT"),destination)
        assert report["status"] == "ALL_REQUESTED_ARTIFACTS_RETAINED"
        assert report["static_resources"]["registers_per_thread"]["status"] == "UNAVAILABLE"
        for kind,item in report["artifacts"].items():
            assert item["sha256"] == old_report["artifacts"][kind]["sha256"]
        for item in report["examples"].values():
            if item["status"] == "AVAILABLE":
                lines = (destination/item["artifact"]).read_text(encoding="utf-8").splitlines()
                assert all(lines[x["line"]-1] == x["text"] for x in item["lines"])
        class MissingAPI:
            @property
            def asm(self): raise RuntimeError("missing API")
            @property
            def metadata(self): raise RuntimeError("missing metadata")
        for label,obj in (("no-object",None),("missing-api",MissingAPI())):
            destination = root / label
            destination.mkdir()
            missing = t.e.export_compiled(obj,destination)
            assert missing["status"] == "COMPILER_OUTPUT_UNAVAILABLE" and len(missing["unavailable"]) == 5
        # A diagnostic-export error must not replace the original checked-call error.
        fail_directory = root / "first-error"
        state = dict(compiler_evidence={})
        def failing_case(*args):
            fail_directory.mkdir()
            (fail_directory/"compiled").mkdir()
            raise ValueError("PRIMARY_CHECK_FAILURE")
        try:
            t.checked_call(None,None,SimpleNamespace(compiled=None),None,t.configurations()[0],"grouped",
                           fail_directory,{},state,case_runner=failing_case)
        except ValueError as error:
            assert str(error) == "PRIMARY_CHECK_FAILURE" and state["failure_capture_errors"]
            rejected.append("secondary-export-error-does-not-mask-primary")
        else:
            raise AssertionError("PRIMARY_FAILURE_NOT_PROPAGATED")
        assert (fail_directory/"compiled").is_dir()
    for name in (*t.NEW_FILES,"33-grouped-compile-cpu-test.py"):
        raw = (HERE/name).read_bytes()
        assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
        if name.endswith(".py"):
            ast.parse(raw.decode("utf-8"))
    assert "torch" not in sys.modules and "triton" not in sys.modules
    report = dict(status="CPU_GROUPED_COMPILE_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",
        plan=t.make_plan(),mock_checked_launches=cases,negative_controls_rejected=rejected,
        accepted32_gate=gate,compiler_export="historical bytes re-exported with equal hashes and literal line excerpts; unavailable fields/object/API stay unavailable",
        snapshot="actual new33 snapshot succeeds on temporary CPU project containing accepted32 source/records; frozen/live/helper mutations rejected",
        limitation="mock launch/provenance/export checks only; no new compiled GPU specialization, timing or hardware-counter evidence",
        source_sha256={name:t.q.sha(HERE/name) for name in (*t.NEW_FILES,"33-grouped-compile-cpu-test.py")})
    output.parent.mkdir(parents=True,exist_ok=True)
    t.h.write_json(output,report)
    print(f"CPU_GROUPED_COMPILE_CONTROLS_PASS negative_controls={len(rejected)} GPU_NOT_RUN REPORT={output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B 33-grouped-compile-cpu-test.py NEW_REPORT_PATH")
    main(Path(sys.argv[1]))
