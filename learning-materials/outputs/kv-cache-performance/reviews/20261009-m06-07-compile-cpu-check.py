"""CPU-only checks of M06-07 capture, export, and preserved driver failures.

Imports the real 24 APIs and the accepted wrapper/checker. Reuses the M06-05
fake tensor support. A CPU scatter stub returns explicitly synthetic compiler
objects; no PyTorch/Triton compilation or GPU execution occurs. Test exports
are temporary and never presented as actual compiler output. Only this review
script and its JSON are permanent. Run with Python -B to avoid bytecode files.
"""
import argparse
import collections
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load(HERE / "reviews/20261009-m06-05-cpu-driver-check.py", "cpu_driver_reuse")
Metadata = collections.namedtuple("Metadata", "shared num_warps specialization")


def synthetic_compiled(parameters):
    """Test data is visibly synthetic and derived from this actual CPU call."""
    label = "CPU_TEST_ONLY_WIDTH_" + str(parameters["WIDTH"])
    assembly = {
        "ttir": "// CPU_TEST_ONLY TTIR " + label + "\n",
        "ttgir": ("CPU_TEST_ONLY TTGIR " + label).encode("utf-8"),
        "llir": bytearray(("CPU_TEST_ONLY LLIR " + label).encode("utf-8")),
        "ptx": "// CPU_TEST_ONLY: intentionally synthetic fixture, not executable PTX\n"
               ".version 0.0\n.target CPU_TEST_ONLY\n.address_size 64\n"
               ".visible .entry " + label + "()\n"
               "// ld.global CPU_TEST_ONLY\n// st.global CPU_TEST_ONLY\n",
        "cubin": b"\x00CPU_TEST_ONLY_BINARY_" + label.encode("ascii") + b"\xff",
    }
    return types.SimpleNamespace(name=label, asm=assembly, n_regs=24,
                                 metadata=Metadata(0, parameters["num_warps"], parameters.copy()))


class CompilingCpuScatter(driver.CpuScatter):
    def __init__(self):
        super().__init__()
        self.actual_result = None
        self.evidence_mode = "full"

    def __getitem__(self, grid):
        def call(kc, vc, indices, k, v, **parameters):
            self.calls.append(dict(grid=list(grid), parameters=parameters.copy()))
            self.scatter(kc, vc, indices, k, v)
            if self.evidence_mode == "none":
                self.actual_result = None
            elif self.evidence_mode == "empty":
                self.actual_result = types.SimpleNamespace(name="CPU_TEST_ONLY_EMPTY", asm={}, metadata=None)
            else:
                self.actual_result = synthetic_compiled(parameters)
            return self.actual_result
        return call


def verify_export(module, obj, directory, expected_status):
    directory.mkdir()
    report = module.export_compiled(obj, directory)
    saved = json.loads((directory / "compiled.json").read_text(encoding="utf-8"))
    assert saved == report
    assert report["status"] == expected_status
    assert report["fresh_compilation"].startswith("NOT_OBSERVED")
    assert "not achieved occupancy or runtime counters" in report["static_resources"]["scope"]
    assert "No one-to-one" in report["interpretation"]
    for kind, artifact in report["artifacts"].items():
        original = obj.asm[kind]
        expected_bytes = original.encode("utf-8") if isinstance(original, str) else bytes(original)
        actual = (directory / artifact["path"]).read_bytes()
        assert actual == expected_bytes
        assert artifact["bytes"] == len(expected_bytes)
        assert artifact["sha256"] == hashlib.sha256(actual).hexdigest()
        if isinstance(original, str):
            assert artifact["lines"] == len(original.splitlines())
        else:
            assert "lines" not in artifact
    if "ptx" not in report["artifacts"]:
        assert report["ptx_excerpt"] == []
        assert not (directory / "compiled/kernel.ptx").exists()
        assert not (directory / "compiled/ptx-excerpt.txt").exists()
        assert "ptx" in report["unavailable"]
    else:
        assert (directory / "compiled/ptx-excerpt.txt").read_text(encoding="utf-8") == "\n".join(report["ptx_excerpt"]) + "\n"
    return report


def run():
    module = load(HERE / "24-inspect-triton-compile.py", "compile_cpu_module")
    support = load(HERE / "22-kv-contract-support.py", "compile_cpu_support")
    source = (driver.ACCEPTED / "triton_store_api.py").read_text(encoding="utf-8-sig")
    module.w.review_api(source)
    cfg = module.CONFIG
    assert cfg in module.c.configurations()
    expected = dict(WIDTH=192, SRC_ROW_STRIDE=256, DST_ROW_STRIDE=256, BLOCK_SIZE=256, num_warps=4)
    stub = CompilingCpuScatter()
    probe = module.CompiledCapture(stub)
    original = module.c.OriginalAdapter(None, support)
    mocks = {"torch": driver.FAKE_TORCH, "kv_contract_support": support,
             "triton_store_mask": types.SimpleNamespace(store_kv_v1=probe)}
    results = []
    with patch.dict(sys.modules, mocks), tempfile.TemporaryDirectory(prefix="m06-07-compile-cpu-") as temp:
        destination = Path(temp)
        api = types.ModuleType("accepted_compile_wrapper_cpu_only")
        exec(compile(source, str(driver.ACCEPTED / "triton_store_api.py"), "exec"), api.__dict__)
        checker = load(driver.ACCEPTED / "check_kv_targets.py", "compile_cpu_checker").check_kv_targets
        before = module.c.fixture(cfg)
        buffers, args = module.c.gpu_packet(driver.FAKE_TORCH, cfg, before)
        probe.reset(True)
        returned = probe[(3,)](*args, **expected)
        assert returned is stub.actual_result and probe.compiled is returned
        assert probe.attempts == [dict(grid=[3], parameters=expected)]
        module.h.verify_backing(before, {n: t.array.copy() for n, t in buffers.items()})
        results.append(dict(name="actual_probe_return_identity", status="CPU_PASS", launch=probe.attempts.copy()))

        # run_case resets the probe; a stale different specialization must not survive.
        stale = synthetic_compiled(dict(expected, WIDTH=64, BLOCK_SIZE=64))
        probe.compiled = stale
        stub.calls.clear()
        driver.EVENTS.clear()
        checked = destination / "checked-call"
        with contextlib.redirect_stdout(io.StringIO()):
            record, after = module.c.run_case(driver.FAKE_TORCH, original, api, probe, checker,
                                              cfg, "learner_triton", checked, before)
        driver.verify_files(checked, before, cfg, "learner_triton")
        assert record["status"] == "PASS"
        assert probe.compiled is stub.actual_result and probe.compiled is not stale
        assert len(stub.calls) == 1 and stub.calls[0] == dict(grid=[3], parameters=expected)
        report = verify_export(module, probe.compiled, destination / "full-export", "PTX_AND_METADATA")
        assert report["metadata"]["specialization"] == expected
        assert report["metadata"]["num_warps"] == 4
        assert report["kernel_name"] == "CPU_TEST_ONLY_WIDTH_192"
        assert set(report["artifacts"]) == {"ttir", "ttgir", "llir", "ptx", "cubin"}
        assert report["static_resources"]["registers_per_thread"] == 24
        assert report["static_resources"]["shared_memory_bytes"] == 0
        results.append(dict(name="executed_specialization_export", status="CPU_PASS",
                            synthetic_fixture=True, metadata=report["metadata"], artifacts=report["artifacts"]))

        binary = types.SimpleNamespace(asm={"cubin": b"CPU_TEST_ONLY_NO_PTX\x00"}, metadata={"shared": 0})
        binary_report = verify_export(module, binary, destination / "binary-only", "BINARY_AND_METADATA")
        assert binary_report["static_resources"]["registers_per_thread"] is None
        results.append(dict(name="missing_ptx_binary_retained", status="CPU_PASS", export_status=binary_report["status"]))
        metadata_only = types.SimpleNamespace(metadata={"num_warps": 4})
        metadata_report = verify_export(module, metadata_only, destination / "metadata-only", "METADATA_OR_IR_ONLY")
        assert metadata_report["artifacts"] == {} and metadata_report["available_asm_keys"] == []
        assert metadata_report["static_resources"]["registers_per_thread"] is None
        assert metadata_report["static_resources"]["shared_memory_bytes"] is None
        results.append(dict(name="missing_asm_metadata_only", status="CPU_PASS", export_status=metadata_report["status"],
                            static_resources=metadata_report["static_resources"]))
        ir_only = types.SimpleNamespace(asm={"ttir": "// CPU_TEST_ONLY_IR\n"})
        ir_report = verify_export(module, ir_only, destination / "ir-only", "METADATA_OR_IR_ONLY")
        assert ir_report["metadata"] is None and list(ir_report["artifacts"]) == ["ttir"]
        results.append(dict(name="ir_without_ptx_or_metadata", status="CPU_PASS", export_status=ir_report["status"]))
        unsupported = types.SimpleNamespace(asm={"ptx": 1234}, metadata={"shared": 0})
        unsupported_report = verify_export(module, unsupported, destination / "unsupported-type", "METADATA_OR_IR_ONLY")
        assert unsupported_report["unavailable"]["ptx"] == "unsupported artifact type: int"
        results.append(dict(name="unsupported_ptx_type_not_synthesized", status="CPU_PASS", unavailable=unsupported_report["unavailable"]))

        # A None return must not silently become invented compiler evidence.
        none_dir = destination / "none-return"
        none_dir.mkdir()
        try:
            module.export_compiled(None, none_dir)
        except RuntimeError as error:
            assert str(error).startswith("NO_COMPILED_RETURN")
            results.append(dict(name="no_compiled_return_rejected", status="CPU_PASS", error=str(error)))
        else:
            raise AssertionError("None compiled return accepted")
        assert not (none_dir / "compiled.json").exists()

        # Successful correctness evidence survives an export failure.
        empty_run = destination / "no-evidence"
        empty_run.mkdir()
        stub.evidence_mode = "empty"
        with contextlib.redirect_stdout(io.StringIO()):
            module.c.run_case(driver.FAKE_TORCH, original, api, probe, checker, cfg,
                              "learner_triton", empty_run / "checked-call", before)
        original_files = {p.name: p.read_bytes() for p in (empty_run / "checked-call").iterdir()}
        try:
            module.export_compiled(probe.compiled, empty_run)
        except RuntimeError as error:
            assert str(error).startswith("NO_COMPILER_EVIDENCE")
            no_evidence_error = str(error)
        else:
            raise AssertionError("No compiler evidence accepted")
        assert not (empty_run / "compiled.json").exists()
        assert original_files == {p.name: p.read_bytes() for p in (empty_run / "checked-call").iterdir()}
        driver.verify_files(empty_run / "checked-call", before, cfg, "learner_triton")
        results.append(dict(name="no_artifacts_or_metadata_preserves_checked_call", status="CPU_PASS",
                            error=no_evidence_error, preserved_files=sorted(original_files)))

        # A failed result must retain call evidence and must not be exported as a
        # successfully checked compiler observation (matching main's call order).
        stub.evidence_mode = "full"
        stub.corrupt_once = True
        failed_call = destination / "failed-call"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.c.run_case(driver.FAKE_TORCH, original, api, probe, checker, cfg,
                                  "learner_triton", failed_call, before)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted CPU output accepted")
        status, submissions, hashes = driver.verify_files(failed_call, before, cfg, "learner_triton", expect_failed=True)
        assert probe.compiled is stub.actual_result and probe.compiled is not None
        assert submissions == [dict(grid=[3], parameters=expected)]
        assert not (failed_call / "compiled.json").exists()
        results.append(dict(name="bad_result_preserves_submission_and_storage", status="CPU_PASS",
                            real_driver_status=status, submissions=submissions, preserved_file_sha256=hashes))
    paths = [HERE / n for n in ("24-inspect-triton-compile.py", "23-check-kv-common.py",
             "22-check-kv-wrapper.py", "22-kv-contract-support.py", "reviews/20261009-m06-05-cpu-driver-check.py")]
    paths += [driver.ACCEPTED / n for n in ("triton_store_api.py", "check_kv_targets.py")]
    return dict(status="CPU_COMPILE_CAPTURE_EXPORT_CHECK_PASS", GPU="NOT_RUN", real_compiler="NOT_RUN",
                new_learner_acceptance="NOT_INFERRED", cases=len(results), results=results,
                method="Real capture/export/run_case APIs; accepted wrapper/checker; fake torch and CPU scatter with synthetic CompiledKernel fixtures; all temporary exports read back and removed",
                source_hashes={str(p.relative_to(HERE)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    result = run()
    result["test_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print("CPU_COMPILE_CAPTURE_EXPORT_CHECK_PASS: " + str(result["cases"]) + " bounded checks; GPU NOT_RUN; real compiler NOT_RUN")
