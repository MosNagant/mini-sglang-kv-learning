"""M06-07: observe one executed Triton specialization, without timing.

Coach probe captures the compiled object returned by the underlying launch.
The accepted learner kernel and public wrapper are read only. Compilation may
reuse a cache; neither fresh compilation nor compilation duration is measured.
"""
import argparse
import dataclasses
import hashlib
import inspect
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compile_common_helpers", HERE / "23-check-kv-common.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
h, w, m, COMMIT = c.h, c.w, c.m, c.COMMIT
CONFIG = dict(batch=3, width=192, capacity=32, layout="separate", gap=64,
              guard=64, rows=[31, 0, 1])


class CompiledCapture(w.LaunchProbe):
    """Record the return from the actual checked launch, including cache hits."""
    def reset(self, allow):
        super().reset(allow)
        self.compiled = None

    def __getitem__(self, grid):
        launch = super().__getitem__(grid)
        def call(*args, **kwargs):
            self.compiled = launch(*args, **kwargs)
            return self.compiled
        return call


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    if hasattr(value, "_asdict"):
        return json_value(value._asdict())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return dict(type=type(value).__name__, representation=repr(value))


def export_compiled(compiled, run):
    """Preserve actual retained artifacts. Never synthesize missing IR/PTX."""
    if compiled is None:
        raise RuntimeError("NO_COMPILED_RETURN: underlying launch returned no compiled object")
    directory = run / "compiled"
    directory.mkdir()
    assembly = getattr(compiled, "asm", None)
    available = list(assembly.keys()) if assembly is not None else []
    artifacts, unavailable = {}, {}
    for kind in ("ttir", "ttgir", "llir", "ptx", "cubin"):
        if kind not in available:
            unavailable[kind] = "not retained by this runtime/cache configuration"
            continue
        value = assembly[kind]
        if not isinstance(value, (str, bytes, bytearray)):
            unavailable[kind] = "unsupported artifact type: " + type(value).__name__
            continue
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        path = directory / ("kernel." + kind)
        path.write_bytes(data)
        artifacts[kind] = dict(path=path.relative_to(run).as_posix(), bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest())
        if isinstance(value, str):
            artifacts[kind]["lines"] = len(value.splitlines())
    metadata = json_value(getattr(compiled, "metadata", None))
    if not artifacts and metadata is None:
        raise RuntimeError("NO_COMPILER_EVIDENCE: no retained artifacts or metadata")
    # Static resource information after loading. These are not hardware counters.
    registers = getattr(compiled, "n_regs", None)
    shared = metadata.get("shared") if isinstance(metadata, dict) else None
    excerpt = []
    if "ptx" in artifacts and isinstance(assembly["ptx"], str):
        lines = assembly["ptx"].splitlines()
        for label, matches in (
            ("header", [s for s in lines if s.strip().startswith((".version", ".target", ".address_size", ".visible .entry"))][:6]),
            ("global_load_examples", [s for s in lines if "ld.global" in s][:6]),
            ("global_store_examples", [s for s in lines if "st.global" in s][:6]),
        ):
            excerpt.extend([label + ":"] + matches)
        (directory / "ptx-excerpt.txt").write_text("\n".join(excerpt) + "\n", encoding="utf-8")
    report = dict(
        status="PTX_AND_METADATA" if "ptx" in artifacts else
               ("BINARY_AND_METADATA" if "cubin" in artifacts else "METADATA_OR_IR_ONLY"),
        kernel_name=getattr(compiled, "name", None), metadata=metadata,
        available_asm_keys=available, artifacts=artifacts, unavailable=unavailable,
        static_resources=dict(registers_per_thread=registers, shared_memory_bytes=shared,
                              scope="loaded specialization metadata; not achieved occupancy or runtime counters"),
        ptx_excerpt=excerpt, fresh_compilation="NOT_OBSERVED; cache reuse is possible",
        interpretation="PTX is virtual ISA; cubin is the device binary. No one-to-one tl.load/store mapping assumed.",
    )
    h.write_json(run / "compiled.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shared-results", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED: run without -O")
    project = args.project.resolve()
    def git(*words):
        return subprocess.check_output(["git", "-C", str(project), *words], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain", "--", "python/minisgl/kernel"):
        raise RuntimeError("PINNED_ORIGINAL_CHANGED: no files rewritten")
    parent = project / "experiments/learning/m06-compile"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = dict(status="PREPARING", lesson="M06-07", commit=COMMIT, config=CONFIG,
                 scope="one checked Triton call and its retained compiler output; no timing",
                 ownership="Accepted learner sources reused; probe, export and recording supplied by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['scope']}", flush=True)
    try:
        hashes = c.snapshot(project, run, git)
        for name in ("24-inspect-triton-compile.py", "24-inspect-triton-compile.sh"):
            shutil.copyfile(HERE / name, run / name)
            hashes[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
        h.write_json(run / "source_hashes.json", hashes)
        h.write_json(run / "source_review.json", dict(
            wrapper=w.review_api((run / "triton_store_api.py").read_text(encoding="utf-8-sig")),
            mask=m.validate_mask_source((run / "triton_store_mask.py").read_text(encoding="utf-8-sig"))))
        import torch
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if triton.__version__ != "3.5.1":
            raise RuntimeError("TRITON_VERSION_CHANGED: inspect compatibility; do not upgrade automatically")
        support = h.load_module(run / "kv_contract_support.py", "kv_contract_support")
        kernel = h.load_module(run / "triton_store_mask.py", "triton_store_mask")
        api = h.load_module(run / "triton_store_api.py", "compile_learner_api")
        checker = h.load_module(run / "check_kv_targets.py", "compile_learner_checker").check_kv_targets
        probe = CompiledCapture(kernel.store_kv_v1)
        api.store_kv_v1 = probe
        state.update(status="RUNNING", torch=torch.__version__, triton=triton.__version__,
                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        h.write_json(run / "status.json", state)
        print(f"GPU={state['gpu']} TRITON={state['triton']}", flush=True)
        before = c.fixture(CONFIG)
        record, _ = c.run_case(torch, c.OriginalAdapter(None, support), api, probe, checker,
                              CONFIG, "learner_triton", run / "checked-call", before)
        report = export_compiled(probe.compiled, run)
        # Preserve the installed runtime files defining the observed object/API.
        runtime_sources = {}
        for label, obj in (("compiler", type(probe.compiled)), ("jit", type(kernel.store_kv_v1))):
            source = inspect.getsourcefile(obj)
            if source is not None and Path(source).is_file():
                dest = run / ("runtime-" + label + ".py")
                shutil.copyfile(source, dest)
                runtime_sources[label] = dict(path=dest.name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        state.update(status="OBSERVATION_READY", correctness=record,
                     launch=probe.attempts, compiler_evidence=report["status"], runtime_sources=runtime_sources)
        print("CHECKED_CONFIGURATION: B=3 D=192 C=32 separate input_gap=64 cache_guard=64")
        print("ACTUAL_LAUNCH:", probe.attempts)
        print("COMPILED_KERNEL_NAME:", report["kernel_name"])
        print("RETAINED_ARTIFACTS:", list(report["artifacts"]))
        print("COMPILE_CACHE: fresh compilation not observed; retained specialization may be reused")
        if report["ptx_excerpt"]:
            print("PTX_EXCERPT_BEGIN\n" + "\n".join(report["ptx_excerpt"]) + "\nPTX_EXCERPT_END")
        else:
            print("PTX_EXCERPT_UNAVAILABLE: inspect retained metadata/artifacts; no synthetic PTX supplied")
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
            if state["status"] == "OBSERVATION_READY":
                raise
    print("COMPILE_OBSERVATION_READY: one actual launch checked; review its saved specialization")
    print("LIMIT: no timing, speedup, fresh-compilation proof, hardware counters or independent mastery claim")


if __name__ == "__main__":
    main()
