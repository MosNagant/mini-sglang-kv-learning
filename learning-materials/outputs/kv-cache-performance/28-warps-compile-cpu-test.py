"""CPU fixture/probe/export controls only; never imports torch or launches GPU."""
import ast
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compile_observation_test",HERE / "28-compare-kv-warps-compile.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def main():
    before = t.fixture()
    good = {n:before["expected_cache" if n == "cache" else n].copy() for n in ("cache","k","v","indices")}
    t.h.verify_backing(before,good)
    rejected = []
    swapped = {n:a.copy() for n,a in good.items()}
    swapped["cache"][[0,8]] = swapped["cache"][[8,0]]
    try:
        t.h.verify_backing(before,swapped)
    except AssertionError:
        rejected.append("wrong_mapping_swap_rows0_and8")
    else:
        raise AssertionError("WRONG_MAPPING_NOT_REJECTED")
    for name,array,index in (
        ("target_K_last","cache",(255,0,127)),("target_V_last","cache",(255,1,127)),
        ("untouched_cache","cache",(1023,1,127)),("input_K_last","k",(255,127)),
        ("input_V_last","v",(255,127)),("indices_last","indices",255),
    ):
        bad = {n:a.copy() for n,a in good.items()}
        bad[array][index] = -99
        try:
            t.h.verify_backing(before,bad)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError("MISSED_BACKING_CORRUPTION: "+name)
    class Tensor:
        def __init__(self,shape,row_stride): self.shape,self.row_stride = shape,row_stride
        def stride(self,axis):
            assert axis == 0
            return self.row_stride
    class Kernel:
        def __init__(self): self.calls = []
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                self.calls.append(dict(grid=grid,parameters=kwargs))
                return self
            return launch
    args = [Tensor((1024,128),256),Tensor((1024,128),256),Tensor((256,),1),
            Tensor((256,128),128),Tensor((256,128),128)]
    for warps in (4,1):
        kernel = Kernel()
        probe = t.CompiledCapture(kernel,warps)
        probe.reset(True)
        params = dict(WIDTH=128,SRC_ROW_STRIDE=128,DST_ROW_STRIDE=256,BLOCK_SIZE=128,num_warps=warps)
        assert probe[(256,)](*args,**params) is kernel and probe.compiled is kernel
        assert probe.attempts == [dict(grid=[256],parameters=params)]
        probe.reset(True)
        assert probe.compiled is None and probe.attempts == []
        try:
            probe[(256,)](*args,**dict(params,num_warps=5-warps))
        except RuntimeError:
            rejected.append("wrong_warps_"+str(warps))
        else:
            raise AssertionError("WRONG_WARP_NOT_BLOCKED")
        assert len(kernel.calls) == 1 and probe.compiled is None
    old = HERE / "logs/m06-compile-results/run-20261009-142759-l12p0a0c"
    old_report = json.loads((old / "compiled.json").read_text(encoding="utf-8"))
    # Re-export existing bytes only to test serialization/excerpts, not new GPU evidence.
    assembly = {kind:(old / item["path"]).read_bytes() for kind,item in old_report["artifacts"].items()}
    with tempfile.TemporaryDirectory(prefix="m07-compile-cpu-") as tmp:
        root = Path(tmp)
        helpers = root / "helpers"
        helpers.mkdir()
        expected_hashes = json.loads((HERE / "logs/m07-warps-timing-results" / t.TIMING_RUN / "source_hashes.json").read_text(encoding="utf-8"))
        for name in t.CURRENT_HELPERS:
            shutil.copyfile(HERE / name,helpers / name)
        t.verify_helpers(helpers,expected_hashes)
        (helpers / "23-check-kv-common.py").write_text("changed",encoding="utf-8")
        try:
            t.verify_helpers(helpers,expected_hashes)
        except RuntimeError as error:
            assert "ACCEPTED_HELPER_CHANGED" in str(error)
            rejected.append("changed_executed_helper")
        else:
            raise AssertionError("CHANGED_HELPER_ACCEPTED")
        arm = root / "old-artifact-export"
        arm.mkdir()
        obj = SimpleNamespace(asm=assembly,metadata=old_report["metadata"],n_regs=18,name="store_kv_v1")
        report = t.export_compiled(obj,arm)
        assert report["status"] == "ALL_REQUESTED_ARTIFACTS_RETAINED"
        assert report["static_resources"]["n_spills_raw"]["status"] == "UNAVAILABLE"
        for kind,item in report["artifacts"].items():
            assert item["sha256"] == old_report["artifacts"][kind]["sha256"]
        for example in report["examples"].values():
            if example["status"] == "AVAILABLE":
                lines = (arm / example["artifact"]).read_text(encoding="utf-8").splitlines()
                assert all(lines[x["line"]-1] == x["text"] for x in example["lines"])
        class MissingAPI:
            @property
            def asm(self): raise RuntimeError("unavailable API")
            @property
            def metadata(self): raise RuntimeError("unavailable metadata")
        for name,obj in (("missing-object",None),("missing-api",MissingAPI()),
                         ("unsupported-artifact",SimpleNamespace(asm={"ptx":123},metadata={"shared":0},n_spills=7))):
            directory = root / name
            directory.mkdir()
            missing = t.export_compiled(obj,directory)
            assert not missing["artifacts"] and len(missing["unavailable"]) == 5
            if name == "unsupported-artifact":
                assert missing["static_resources"]["n_spills_raw"]["value"] == 7
                assert "unit not inferred" in missing["static_resources"]["n_spills_raw"]["units"]
        try:
            t.export_compiled(None,arm)
        except FileExistsError:
            rejected.append("existing_compiler_export_not_overwritten")
        else:
            raise AssertionError("EXISTING_ARTIFACTS_OVERWRITTEN")
    for name,expected in t.ACCEPTED_RUNNERS.items():
        assert t.q.sha(HERE / name) == expected
    for name in ("28-compare-kv-warps-compile.py","28-warps-compile-cpu-test.py"):
        ast.parse((HERE / name).read_text(encoding="utf-8"))
    shell = (HERE / "28-compare-kv-warps-compile.sh").read_bytes()
    assert b"\r" not in shell and not shell.startswith(b"\xef\xbb\xbf")
    assert "torch" not in sys.modules and "triton" not in sys.modules
    print(json.dumps(dict(status="CPU_COMPILE_HARNESS_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",
                         config={k:v for k,v in t.CONFIG.items() if k != "rows"},indices="0..255",negative_controls_rejected=rejected,
                         compiler_export="old retained bytes re-exported with identical hashes; original-line excerpts verified; missing fields/API marked unavailable",
                         source_sha256={name:t.q.sha(HERE/name) for name in ("28-compare-kv-warps-compile.py","28-compare-kv-warps-compile.sh","28-warps-compile-cpu-test.py")}),indent=2))


if __name__ == "__main__":
    main()
