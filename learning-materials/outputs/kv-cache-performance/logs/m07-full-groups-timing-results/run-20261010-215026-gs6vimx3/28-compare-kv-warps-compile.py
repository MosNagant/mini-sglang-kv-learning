"""M07: retain actual 4-warp / 1-warp compiler output after checked GPU calls.

B256 D128 C1024, interleaved FP16, contiguous inputs, unique int32 indices.
No timing, cache deletion or fresh-compilation assertion. Missing runtime fields
and artifacts are reported unavailable, never inferred from prior runs.
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

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("warps_compile_timing_helpers", HERE / "27-compare-kv-warps.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
q, c, h, w, COMMIT = p.q, p.c, p.h, p.w, p.COMMIT
d = c.helper("warps_compile_export_helpers", "24-inspect-triton-compile.py")
TIMING_RUN = "run-20261010-121112-p7ip91zk"
ACCEPTED_RUNNERS = {
    "26-check-kv-warps.py": "0d60879bf1f94d9a2192ddf0c093a059b8f459961d22397dc847c375da779da4",
    "27-compare-kv-warps.py": "edb64af3a884ffbf7d727dae944fcd181bc9e975c886ce21f3057a27fab42aaa",
}
CONFIG = dict(batch=256, width=128, capacity=1024, layout="interleaved", gap=0,
              guard=0, rows=list(range(256)))
KINDS = ("ttir", "ttgir", "llir", "ptx", "cubin")
CURRENT_HELPERS = tuple(q.HELPERS) + ("25-compare-kv-v1.py","27-compare-kv-warps.py","27-compare-kv-warps.sh")


class CompiledCapture(q.LaunchProbe):
    def reset(self, allow):
        super().reset(allow)
        self.compiled = None

    def __getitem__(self, grid):
        launch = super().__getitem__(grid)
        def call(*args, **kwargs):
            self.compiled = launch(*args, **kwargs)
            return self.compiled
        return call


def fixture():
    """Independent bounded CPU oracle, including every untouched cache row."""
    b,d,capa = (CONFIG[k] for k in ("batch", "width", "capacity"))
    k = np.random.default_rng(20261010).integers(0,1024,(b,d),dtype=np.int32).astype(np.float16)
    k[:,0] = np.arange(b,dtype=np.float16)  # Every input row is distinguishable.
    before = dict(cache=np.full((capa,2,d),-1,dtype=np.float16), k=k,
                  v=k+np.float16(1024), indices=np.arange(b,dtype=np.int32))
    expected = before["cache"].copy()
    for row,position in enumerate(before["indices"]):
        expected[position,0,:] = before["k"][row]
        expected[position,1,:] = before["v"][row]
    before["expected_cache"] = expected
    return before


def unavailable(reason):
    return dict(status="UNAVAILABLE", reason=reason)


def optional_attr(obj, name):
    try:
        value = getattr(obj,name)
        return value, None if value is not None else "field is None"
    except Exception as error:
        return None, type(error).__name__ + ": " + str(error)


def resource(value, reason, source, units):
    if reason is not None:
        return dict(unavailable(reason), source=source, units=units)
    return dict(status="AVAILABLE", value=d.json_value(value), source=source, units=units)


def export_compiled(compiled, arm_directory):
    """Retain actual bytes and literal examples with original line numbers."""
    directory = arm_directory / "compiled"
    directory.mkdir()
    report = dict(status="COMPILER_OUTPUT_UNAVAILABLE", artifacts={}, unavailable={},
                  available_asm_keys=[], metadata=unavailable("no compiled object returned"),
                  static_resources={}, examples={},
                  fresh_compilation="NOT_OBSERVED; runtime/cache reuse is possible; cache not deleted",
                  interpretation="Static loaded-specialization data, not runtime occupancy/counters. PTX is virtual ISA; no one-to-one source mapping or performance cause inferred.")
    if compiled is None:
        report["unavailable"] = {kind:"launch returned no compiled object" for kind in KINDS}
        h.write_json(arm_directory / "compiled.json",report)
        return report
    assembly, asm_error = optional_attr(compiled,"asm")
    try:
        available = list(assembly.keys()) if asm_error is None else []
    except Exception as error:
        available, asm_error = [], type(error).__name__ + ": " + str(error)
    report["available_asm_keys"] = [str(k) for k in available]
    text_artifacts = {}
    for kind in KINDS:
        if kind not in available:
            report["unavailable"][kind] = asm_error or "not retained by this runtime/cache configuration"
            continue
        try:
            value = assembly[kind]
        except Exception as error:
            report["unavailable"][kind] = type(error).__name__ + ": " + str(error)
            continue
        if not isinstance(value,(str,bytes,bytearray)):
            report["unavailable"][kind] = "unsupported artifact type: " + type(value).__name__
            continue
        data = value.encode("utf-8") if isinstance(value,str) else bytes(value)
        path = directory / ("kernel."+kind)
        path.write_bytes(data)
        report["artifacts"][kind] = dict(path=path.relative_to(arm_directory).as_posix(),
                                        bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if kind != "cubin":
            try:
                text_artifacts[kind] = data.decode("utf-8")
                report["artifacts"][kind]["lines"] = len(text_artifacts[kind].splitlines())
            except UnicodeDecodeError:
                report.setdefault("text_decode_unavailable",{})[kind] = "not UTF-8; original bytes retained"
    raw_metadata, metadata_error = optional_attr(compiled,"metadata")
    try:
        metadata = d.json_value(raw_metadata) if metadata_error is None else None
    except Exception as error:
        metadata,metadata_error = None,type(error).__name__+": "+str(error)
    report["metadata"] = resource(metadata,metadata_error,"compiled.metadata","runtime metadata fields")
    name, name_error = optional_attr(compiled,"name")
    report["kernel_name"] = resource(name,name_error,"compiled.name","string")
    regs, regs_error = optional_attr(compiled,"n_regs")
    spills, spills_error = optional_attr(compiled,"n_spills")
    shared = metadata.get("shared") if isinstance(metadata,dict) else None
    report["static_resources"] = dict(
        registers_per_thread=resource(regs,regs_error,"compiled.n_regs","registers per thread"),
        shared_memory_bytes=resource(shared,None if shared is not None else "metadata.shared unavailable","compiled.metadata.shared","bytes"),
        n_spills_raw=resource(spills,spills_error,"compiled.n_spills","runtime-reported raw field; unit not inferred"),
        scope="static loaded specialization; not achieved occupancy, spill traffic, bandwidth or runtime hardware counters")
    selectors = {
        "ttgir_layout":("ttgir",lambda line:"#triton_gpu." in line or "#ttg." in line or "sizePerThread" in line or "threadsPerWarp" in line),
        "ttir_load_store":("ttir",lambda line:"tt.load" in line or "tt.store" in line),
        "ptx_header":("ptx",lambda line:line.strip().startswith((".version",".target",".address_size",".visible .entry"))),
        "ptx_global_load":("ptx",lambda line:"ld.global" in line),
        "ptx_global_store":("ptx",lambda line:"st.global" in line),
    }
    for label,(kind,select) in selectors.items():
        matches = [dict(line=index,text=line) for index,line in enumerate(text_artifacts.get(kind,"").splitlines(),1) if select(line)][:8]
        report["examples"][label] = (dict(status="AVAILABLE",artifact="compiled/kernel."+kind,lines=matches)
                                      if matches else unavailable("no matching lines in the retained UTF-8 artifact"))
    report["status"] = "ALL_REQUESTED_ARTIFACTS_RETAINED" if len(report["artifacts"]) == len(KINDS) else "PARTIAL_COMPILER_EVIDENCE"
    if not report["artifacts"] and metadata_error is not None:
        report["status"] = "COMPILER_OUTPUT_UNAVAILABLE"
    h.write_json(arm_directory / "compiled.json",report)
    return report


def verify_helpers(directory,expected_hashes):
    for name in CURRENT_HELPERS:
        if q.sha(directory / name) != expected_hashes[name]:
            raise RuntimeError("ACCEPTED_HELPER_CHANGED: " + name)


def print_observation(arm,report):
    print("STATIC_RESOURCES arm="+arm+" "+json.dumps(report["static_resources"]),flush=True)
    for label in ("ttgir_layout","ptx_global_load","ptx_global_store"):
        example = report["examples"].get(label,unavailable("no compiled object"))
        if example["status"] == "AVAILABLE":
            for line in example["lines"][:3]:
                print(f"ACTUAL_EXCERPT arm={arm} kind={label} line={line['line']}: {line['text']}",flush=True)
        else:
            print(f"EXCERPT_UNAVAILABLE arm={arm} kind={label}: {example['reason']}",flush=True)


def snapshot(project,run):
    for name,expected in ACCEPTED_RUNNERS.items():
        if q.sha(HERE/name) != expected:
            raise RuntimeError("ACCEPTED_RUNNER_CHANGED: " + name)
    correctness = p.accepted_gate(project / "experiments/learning/m07-warps" / p.ACCEPTED_RUN)
    accepted = project / "experiments/learning/m07-warps-timing" / TIMING_RUN
    status = json.loads((accepted / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COLLECTION_COMPLETE" or status.get("commit") != COMMIT:
        raise RuntimeError("ACCEPTED_TIMING_STATUS_REQUIRED")
    if p.summarize(status["results"]) != status["summary"]:
        raise RuntimeError("ACCEPTED_TIMING_SUMMARY_CHANGED")
    p.verify_snapshot(accepted,project)
    old_hashes = json.loads((accepted / "source_hashes.json").read_text(encoding="utf-8"))
    for name,expected in ACCEPTED_RUNNERS.items():
        if old_hashes.get(name) != expected:
            raise RuntimeError("ACCEPTED_TIMING_RUNNER_MISMATCH: " + name)
    verify_helpers(HERE,old_hashes)
    q.check_pinned(project / "experiments")
    if q.sha(project / "experiments" / q.CANDIDATE) != p.CANDIDATE_HASH:
        raise RuntimeError("LIVE_CANDIDATE_CHANGED: preserve and review before observing")
    for name in old_hashes:
        destination = run / name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(accepted / name,destination)
    for name in ("24-inspect-triton-compile.py","28-compare-kv-warps-compile.py","28-compare-kv-warps-compile.sh"):
        shutil.copyfile(HERE / name,run / name)
    shutil.copyfile(HERE / "lessons/m07-03-warps-compile.md",run / "compiler-observation-lesson.md")
    h.write_json(run / "accepted_source_gates.json",dict(correctness=correctness,
        timing=dict(run=TIMING_RUN,status_sha256=q.sha(accepted / "status.json"),
                    source_hashes_sha256=q.sha(accepted / "source_hashes.json"),status="ACCEPTED_SOURCE_AND_SUMMARY_GATE_PASS"),
        live_candidate_sha256=p.CANDIDATE_HASH,accepted_runners=ACCEPTED_RUNNERS))
    hashes = {f.relative_to(run).as_posix():q.sha(f) for f in run.rglob("*") if f.is_file()}
    h.write_json(run / "source_hashes.json",hashes)
    p.verify_snapshot(run,project)


def runtime_sources(compiled,kernel,run):
    records = {}
    for label,obj in (("compiler",type(compiled) if compiled is not None else None),("jit",type(kernel))):
        try:
            source = inspect.getsourcefile(obj) if obj is not None else None
            if source is None or not Path(source).is_file():
                records[label] = unavailable("runtime source file not available")
                continue
            destination = run / ("runtime-"+label+".py")
            shutil.copyfile(source,destination)
            records[label] = dict(status="AVAILABLE",path=destination.name,sha256=q.sha(destination))
        except Exception as error:
            records[label] = unavailable(type(error).__name__+": "+str(error))
    h.write_json(run / "runtime-sources.json",records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--shared-results",type=Path,required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED")
    project = args.project.resolve()
    commit = subprocess.check_output(["git","-C",str(project),"rev-parse","HEAD"],text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("PINNED_PROJECT_COMMIT_CHANGED")
    parent = project / "experiments/learning/m07-warps-compile"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 compiler observation",config=CONFIG,arms=q.ARMS,
                 scope="two independent checked B256/D128 calls and actual retained specializations; no timing",
                 fixture="NumPy default_rng seed20261010; bounded integers and first-column row IDs; same layout/shape as27, different input values",
                 correctness={},compiler_evidence={},ownership="accepted learner parameter change; coach observation harness")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['scope']}",flush=True)
    try:
        snapshot(project,run)
        import torch
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if (torch.__version__,torch.version.cuda,triton.__version__) != ("2.9.1+cu128","12.8","3.5.1"):
            raise RuntimeError("FIXED_ENVIRONMENT_CHANGED: review without automatic upgrade")
        h.load_module(run / "kv_contract_support.py","kv_contract_support")
        kernel = h.load_module(run / "triton_store_mask.py","triton_store_mask").store_kv_v1
        apis = dict(frozen_v1=h.load_module(run / "triton_store_api.py","compile_frozen_v1"),
                    candidate=h.load_module(run / q.CANDIDATE,"compile_candidate"))
        checker = h.load_module(run / "check_kv_targets.py","compile_checker").check_kv_targets
        state.update(status="RUNNING",commit=COMMIT,torch=torch.__version__,triton=triton.__version__,
                     cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                     device_properties=str(torch.cuda.get_device_properties(0)))
        before,results = fixture(),[]
        for arm,warps in q.ARMS.items():
            directory = run / arm
            state["current_arm"] = arm
            h.write_json(run / "status.json",state)
            probe = CompiledCapture(kernel,warps)
            apis[arm].store_kv_v1 = probe
            try:
                record,after = q.run_case(torch,apis[arm],probe,checker,CONFIG,arm,directory,before)
            except BaseException:
                try:
                    state["compiler_evidence"][arm] = export_compiled(probe.compiled,directory)
                except Exception as capture_error:
                    state.setdefault("failure_capture_errors",[]).append(dict(arm=arm,error=str(capture_error)))
                raise
            report = export_compiled(probe.compiled,directory)
            state["correctness"][arm] = record
            state["compiler_evidence"][arm] = report
            results.append(after)
            state.setdefault("runtime_sources",{})[arm] = runtime_sources(probe.compiled,kernel,directory)
            print(f"CHECKED_ARM={arm} ACTUAL_LAUNCH={probe.attempts} COMPILER_OUTPUT={report['status']}",flush=True)
            print_observation(arm,report)
        for name in ("cache","k","v","indices"):
            np.testing.assert_array_equal(results[0][name],results[1][name],err_msg="PAIR_"+name)
        p.verify_snapshot(run,project)
        state.update(status="OBSERVATION_READY",current_arm=None,pair="BOTH_MATCH_INDEPENDENT_REFERENCE_AND_EACH_OTHER")
        print("WARPS_COMPILE_OBSERVATION_READY: two checked calls; inspect each arm's compiled.json and retained files")
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json",state)
        p.export_run(run,args.shared_results,state["status"] == "OBSERVATION_READY")
    print("LIMIT: no timing, fresh-compilation proof, occupancy/hardware counters, vector-width assumption or performance-cause claim")


if __name__ == "__main__":
    main()
