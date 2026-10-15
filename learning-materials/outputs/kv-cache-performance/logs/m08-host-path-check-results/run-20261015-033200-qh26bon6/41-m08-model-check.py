"""Three fresh real-model workers: original repeat, original, and explicit Triton.

Diagnostic only. Every store checks the entire initialized KV allocation and
readonly input allocations. Cross-process comparisons use exact bytes, no tolerance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
HELPER = "39-m08-model-smoke.py"
ADAPTER = "40-m08-kv-backends.py"
HELPER_SHA256 = "b176e1dc214597bed43862f76bffd5bbe6f9d27a2f680d38e5e4ee9749af655e"
ADAPTER_SHA256 = "cc4305af9cf89985f494b90f990070c5c042a328722866aad4d67a5353f81c15"
ARMS = ("original_repeat", "original", "triton")
ENGINE_OPTIONS = {
    "max_running_req": 1, "attention_backend": "fi", "page_size": 1,
    "num_page_override": 512, "max_seq_len_override": 256,
    "max_extend_tokens": 256, "cuda_graph_bs": [], "memory_ratio": 0.3,
    "use_dummy_weight": False, "use_pynccl": False, "cache_type": "naive",
}
PROMPT = ("This diagnostic prompt exercises the key value cache during language model generation. " * 20)
INPUT_TOKENS = 128
OUTPUT_TOKENS = 32
SENTINEL = -7.0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h = load_module("m08_smoke_helpers", HERE / HELPER)
if h.sha256(HERE / HELPER) != HELPER_SHA256:
    raise ValueError("The reviewed 39 helper has changed")


def validate_request(request):
    if request["engine_options"] != ENGINE_OPTIONS or request["max_new_tokens"] != OUTPUT_TOKENS:
        raise ValueError("Unexpected model-check execution options")
    ids = request["input_token_ids"]
    if len(ids) != INPUT_TOKENS or any(type(i) is not int or not 0 <= i < request["model"]["config"]["vocab_size"] for i in ids):
        raise ValueError("Expected exactly 128 valid real-tokenizer token IDs")
    if request["helper_sha256"] != HELPER_SHA256:
        raise ValueError("Helper hash mismatch")
    if request["adapter_sha256"] != ADAPTER_SHA256:
        raise ValueError("Reviewed adapter hash mismatch")


def verify_snapshot(run, request):
    records = json.loads((run / "source-manifest.json").read_text(encoding="utf-8"))
    for relative, digest in records.items():
        if h.sha256(run / "source" / relative) != digest:
            raise ValueError(f"Saved upstream source changed: {relative}")
    for name, digest in [(HELPER, request["helper_sha256"]), (ADAPTER, request["adapter_sha256"]),
                         (Path(__file__).name, request["runner_sha256"])]:
        if h.sha256(run / name) != digest:
            raise ValueError(f"Saved diagnostic source changed: {name}")


def verify_model_metadata(model):
    for name, digest in model["metadata_sha256"].items():
        if Path(name).name != name or h.sha256(Path(model["path"]) / name) != digest:
            raise ValueError(f"Model configuration/tokenizer metadata changed: {name}")


def check_launch(selected, before_count, backend, tensors):
    k_cache, v_cache, indices, k, v = tensors
    b, c = int(k.shape[0]), int(k_cache.shape[0])
    d = int(k.numel() // b)
    expected = {"backend": backend, "B": b, "C": c, "D": d,
                "input_k_stride": k.stride(0), "input_v_stride": v.stride(0),
                "cache_k_stride": k_cache.stride(0), "cache_v_stride": v_cache.stride(0),
                "grid": [b] if backend == "triton" else None,
                "BLOCK_D": 1 << (d - 1).bit_length() if backend == "triton" else None,
                "num_warps": 4 if backend == "triton" else None}
    if selected.launch_count != before_count + 1 or selected.call_count != selected.launch_count:
        raise RuntimeError("Selected backend did not record exactly one successful real launch")
    if not isinstance(selected.last_launch, dict) or any(selected.last_launch.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Actual backend launch metadata disagrees with tensor metadata")
    if backend == "triton" and selected.compiled_kernel is None:
        raise RuntimeError("Triton did not return the real CompiledKernel")
    return dict(selected.last_launch)


def compare_npz(left, right):
    import numpy as np
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise AssertionError(f"Array names differ: {left.name}")
        arrays = []
        for key in sorted(a.files):
            x, y = a[key], b[key]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes(order="C") != y.tobytes(order="C"):
                raise AssertionError(f"Bitwise array mismatch: {left.name}:{key}")
            arrays.append({"name": key, "shape": list(x.shape), "dtype": str(x.dtype), "bytes": x.nbytes})
    return arrays


def compare_arms(left, right):
    a = json.loads((left / "result.json").read_text(encoding="utf-8"))
    b = json.loads((right / "result.json").read_text(encoding="utf-8"))
    if a["status"] != "MODEL_ARM_CHECKED" or b["status"] != "MODEL_ARM_CHECKED":
        raise AssertionError("Only fully checked workers may be compared")
    if a["raw_generated_token_ids"] != b["raw_generated_token_ids"]:
        raise AssertionError("Raw generated token sequences differ")
    if a["input_token_ids"] != b["input_token_ids"]:
        raise AssertionError("Input workloads differ")
    files = ["layer0-prefill.npz", "layer0-decode.npz", "sampled-logits.npz", "final-cache.npz"]
    return {"left": left.name, "right": right.name, "comparison": "dtype + shape + exact C-order bytes",
            "tokens_equal": True, "arrays": {name: compare_npz(left / name, right / name) for name in files}}


def worker(run, arm):
    import numpy as np
    request = json.loads((run / "request.json").read_text(encoding="utf-8"))
    validate_request(request)
    verify_snapshot(run, request)
    verify_model_metadata(request["model"])
    out = run / arm
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(run / "source/python"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", MINISGL_DISABLE_OVERLAP_SCHEDULING="1")
    import torch
    import minisgl.kernel as kernel_api
    import minisgl.kernel.store as store_module
    from minisgl.core import SamplingParams, get_global_ctx
    from minisgl.kvcache.mha_pool import MHAKVCache
    from minisgl.llm import LLM
    adapter = load_module("checked_m08_adapter", run / ADAPTER)
    original_store, original_pool = store_module.store_cache, MHAKVCache.store_kv
    if kernel_api.store_cache is not original_store:
        raise RuntimeError("Public alias was already replaced")
    if Path(inspect.getsourcefile(original_store)).resolve() != run / "source/python/minisgl/kernel/store.py":
        raise RuntimeError("Wrong upstream module imported")
    backend = "triton" if arm == "triton" else "original"
    original_calls = 0

    def counted_original(*args, **kwargs):
        nonlocal original_calls
        if backend == "triton":
            raise RuntimeError("FORBIDDEN_ORIGINAL_FALLBACK_IN_TRITON_ARM")
        result = original_store(*args, **kwargs)
        original_calls += 1
        return result

    # Factory imports JIT support but performs no CUDA launch/device initialization.
    selected = adapter.make_store_cache(backend, original_store=counted_original, observe=True)
    active_layer = None
    records, raw_tokens, logits_saved = [], [], {}
    trace_file = (out / "store-checks.jsonl").open("x", encoding="utf-8")
    last_phase = None
    sample_count = 0

    def emit(record):
        trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        trace_file.flush()

    def save_arrays(path, **arrays):
        np.savez_compressed(path, **{name: tensor.detach().cpu().numpy() for name, tensor in arrays.items()})

    def storage_bytes(tensor):
        storage = tensor.untyped_storage()
        return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),), (1,))

    def equal_bits(left, right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))

    def checked_pool(pool, k, v, out_loc, layer_id):
        nonlocal active_layer
        if active_layer is not None:
            raise RuntimeError("Nested KV call")
        active_layer = int(layer_id)
        try:
            return original_pool(pool, k, v, out_loc, layer_id)
        finally:
            active_layer = None

    def checked_store(k_cache, v_cache, indices, k, v):
        nonlocal last_phase
        if active_layer is None:
            raise RuntimeError("KV call without layer attribution")
        context = get_global_ctx()
        batch, pool = context.batch, context.kv_cache
        last_phase = batch.phase
        ids = indices.detach().cpu().tolist()
        c, b, d = int(k_cache.shape[0]), int(k.shape[0]), int(k.numel() // k.shape[0])
        if len(ids) != b or len(set(ids)) != b or any(i < 0 or i >= c for i in ids):
            raise RuntimeError("Actual model indices violate the unique/in-range contract")
        cache = pool._kv_buffer
        cache_pointer = cache.untyped_storage().data_ptr()
        readonly = {}
        for name, tensor in [("k", k), ("v", v), ("indices", indices)]:
            pointer = tensor.untyped_storage().data_ptr()
            if pointer == cache_pointer:
                raise RuntimeError("Readonly input aliases the writable cache")
            if pointer not in readonly:
                view = storage_bytes(tensor)
                readonly[pointer] = (name, view, view.clone())
        before = cache.clone()
        k_before, v_before, ids_before = k.clone(), v.clone(), indices.clone()
        expected = before.clone()
        long_ids = ids_before.to(torch.int64)
        expected[0, active_layer].view(c, d).index_copy_(0, long_ids, k_before.view(b, d))
        expected[1, active_layer].view(c, d).index_copy_(0, long_ids, v_before.view(b, d))
        record = {"call_id": len(records), "layer": active_layer, "phase": batch.phase,
                  "tokens_B": b, "D": d, "indices": ids,
                  "positions": batch.positions.detach().cpu().tolist(),
                  "input_token_ids": batch.input_ids.detach().cpu().tolist(),
                  "tensors": {name: h.tensor_metadata(t) for name, t in [("k_cache", k_cache), ("v_cache", v_cache), ("k", k), ("v", v), ("indices", indices)]},
                  "readonly_allocations": len(readonly), "status": "BEFORE_BACKEND"}
        records.append(record)
        emit(record)
        selected_sample = active_layer == 0 and batch.phase not in {r["phase"] for r in records[:-1] if r["layer"] == 0}
        try:
            previous = selected.launch_count
            result = selected(k_cache, v_cache, indices, k, v)
            record["actual_launch"] = check_launch(selected, previous, backend, (k_cache, v_cache, indices, k, v))
            torch.cuda.synchronize()
            if not equal_bits(cache, expected):
                raise AssertionError("Entire KV allocation differs from independent index-copy expectation; includes targets, untouched rows, other layers, and dummy page")
            if any(not torch.equal(view, old) for _, view, old in readonly.values()):
                raise AssertionError("Readonly K/V/indices backing bytes changed")
            if selected_sample:
                save_arrays(out / f"layer0-{batch.phase}.npz", k=k_before, v=v_before, indices=ids_before,
                            before_k=before[0, active_layer], before_v=before[1, active_layer],
                            expected_k=expected[0, active_layer], expected_v=expected[1, active_layer],
                            after_k=cache[0, active_layer], after_v=cache[1, active_layer])
            record["status"] = "FULL_BACKING_AND_READONLY_CHECKED"
            if backend == "triton":
                record["compiled_kernel_type"] = type(selected.compiled_kernel).__name__
                record["compiled_kernel_hash"] = str(getattr(selected.compiled_kernel, "hash", "unavailable"))
            emit(record)
            return result
        except BaseException:
            record["status"] = "FAILED"
            record["error"] = traceback.format_exc()
            try:
                emit(record)
                save_arrays(out / f"failed-call-{record['call_id']:04d}.npz", before_cache=before,
                            expected_cache=expected, actual_cache=cache, k_before=k_before,
                            v_before=v_before, indices_before=ids_before, k_after=k, v_after=v, indices_after=indices,
                            **{f"readonly_before_{name}": old for name, _, old in readonly.values()},
                            **{f"readonly_after_{name}": view for name, view, _ in readonly.values()})
            except BaseException:
                record["diagnostic_save_error"] = traceback.format_exc()
            raise

    class ObservedLLM(LLM):
        def offline_send_result(self, reply):
            for msg in reply:
                raw_tokens.append({"uid": msg.uid, "token_id": int(msg.next_token), "finished": msg.finished})
            h.save_json(out / "raw-output-tokens.json", raw_tokens)
            return super().offline_send_result(reply)

    llm, primary = None, None
    result = {"status": "FAILED", "arm": arm, "backend": backend, "input_token_ids": request["input_token_ids"],
              "scope": "new 128-input/32-output workload; eager diagnostic only, no timing",
              "authorship": "coach implemented adapter and diagnostic; no independent learner credit"}
    try:
        kernel_api.store_cache = checked_store
        MHAKVCache.store_kv = checked_pool
        llm = ObservedLLM(request["model"]["path"], dtype=torch.float16, **ENGINE_OPTIONS)
        if llm.engine.graph_runner.max_graph_bs != 0:
            raise RuntimeError("Graph must remain disabled for per-call checks")
        state = llm.engine.model.state_dict()
        if not state or any(t.device.type != "cuda" or t.dtype != torch.float16 for t in state.values()):
            raise RuntimeError("Real model weights not materialized on CUDA in FP16")
        with torch.cuda.stream(llm.engine.stream):
            llm.engine.kv_cache._kv_buffer.fill_(SENTINEL)
        torch.cuda.synchronize()
        result["cache_allocation"] = h.tensor_metadata(llm.engine.kv_cache._kv_buffer)
        result["cache_initial_value"] = SENTINEL
        result["environment"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(0), "python": sys.version,
                                 "capability": list(torch.cuda.get_device_capability(0))}
        h.save_json(out / "initialization.json", result)
        original_sample = llm.engine.sampler.sample

        def checked_sample(logits, args):
            nonlocal sample_count
            expected_phase = "prefill" if sample_count == 0 else "decode"
            if last_phase != expected_phase or len(records) != (sample_count + 1) * 28:
                raise RuntimeError("Sampler/store phase attribution mismatch")
            if not bool(torch.isfinite(logits).all().item()):
                raise AssertionError("Non-finite real model logits")
            if sample_count in (0, 1, OUTPUT_TOKENS - 1):
                name = {0: "first_prefill", 1: "first_decode", OUTPUT_TOKENS - 1: "last_decode"}[sample_count]
                logits_saved[name] = logits.detach().cpu().numpy().copy()
                np.savez_compressed(out / "sampled-logits.npz", **logits_saved)
            sample_count += 1
            return original_sample(logits, args)

        llm.engine.sampler.sample = checked_sample
        outputs = llm.generate([request["input_token_ids"]], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=OUTPUT_TOKENS))
        torch.cuda.synchronize()
        h.save_json(out / "outputs.json", outputs)
        save_arrays(out / "final-cache.npz", cache=llm.engine.kv_cache._kv_buffer)
        expected_counts = Counter({(phase, layer): count for layer in range(28)
                                   for phase, count in [("prefill", 1), ("decode", OUTPUT_TOKENS - 1)]})
        if Counter((r["phase"], r["layer"]) for r in records) != expected_counts or any(r["status"] != "FULL_BACKING_AND_READONLY_CHECKED" for r in records):
            raise AssertionError("Missing per-layer real store checks")
        if selected.launch_count != 28 * OUTPUT_TOKENS or sample_count != OUTPUT_TOKENS:
            raise AssertionError("Real dispatch/sampling count mismatch")
        if original_calls != (0 if backend == "triton" else selected.launch_count):
            raise AssertionError("Original/candidate routing mismatch")
        if len(raw_tokens) != OUTPUT_TOKENS or any(t["uid"] != 0 or not 0 <= t["token_id"] < request["model"]["config"]["vocab_size"] for t in raw_tokens):
            raise AssertionError("Incomplete or invalid raw generated token sequence")
        for item in request["model"]["weights"]:
            stat = (Path(request["model"]["path"]) / item["name"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise RuntimeError("Checkpoint changed during the run")
        verify_model_metadata(request["model"])
        result.update(status="MODEL_ARM_CHECKED", raw_generated_token_ids=[t["token_id"] for t in raw_tokens],
                      store_calls=len(records), full_cache_checks=len(records),
                      readonly_allocation_checks=sum(r["readonly_allocations"] for r in records),
                      original_calls=original_calls, selected_launch_count=selected.launch_count,
                      sampled_logit_names=sorted(logits_saved), phase_store_counts={"prefill": 28, "decode": 868})
        result["artifacts"] = {p.name: {"sha256": h.sha256(p), "bytes": p.stat().st_size}
                               for p in sorted(out.iterdir()) if p.suffix in (".npz", ".jsonl")}
    except BaseException:
        primary = traceback.format_exc()
        result["error"] = primary
        (out / "error.txt").write_text(primary, encoding="utf-8")
        raise
    finally:
        kernel_api.store_cache = original_store
        MHAKVCache.store_kv = original_pool
        try:
            trace_file.close()
            if llm is not None:
                llm.shutdown()
        except BaseException:
            result["shutdown_error"] = traceback.format_exc()
            if primary is None:
                result["status"] = "FAILED"
                h.save_json(out / "result.json", result)
                raise
        h.save_json(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home() / "projects/mini-sglang-kv")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--shared-results", type=Path, default=HERE / "logs/m08-model-check-results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-run", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_run:
        if args.arm is None:
            parser.error("--worker-run requires --arm")
        worker(args.worker_run.resolve(), args.arm)
        return
    if args.dry_run:
        print(json.dumps({"status": "PLAN_ONLY_NO_MODEL_OR_GPU_RUN", "arms": ARMS, "fresh_processes": 3,
                          "input_tokens": INPUT_TOKENS, "new_tokens": OUTPUT_TOKENS,
                          "engine_options": ENGINE_OPTIONS, "store_checks_per_worker": 896,
                          "comparison": "bitwise token/cache/logits; no tolerance", "model_path": str(args.model_path)}, indent=2))
        return
    project = args.project.expanduser().resolve(strict=True)
    results_root = project / "experiments/m08-model-check-results"
    results_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-"), dir=results_root))
    print("RUN_DIRECTORY=" + str(run), flush=True)
    result = {"status": "FAILED", "arms_paths": {arm: arm for arm in ARMS},
              "diagnostic_not_timing": True, "comparison_policy": "exact bytes, no tolerance"}
    primary = None
    try:
        h.snapshot_source(project, run)
        for path in [Path(__file__), HERE / HELPER, HERE / ADAPTER]:
            shutil.copyfile(path, run / path.name)
        shutil.copyfile(HERE / "41-m08-model-check.sh", run / "41-m08-model-check.sh")
        model = h.inspect_model(args.model_path, args.model_revision)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
        input_ids = tokenizer.encode(PROMPT)[:INPUT_TOKENS]
        request = {"model": model, "input_token_ids": input_ids, "prompt_source_text": PROMPT,
                   "max_new_tokens": OUTPUT_TOKENS, "engine_options": ENGINE_OPTIONS,
                   "adapter_sha256": h.sha256(run / ADAPTER), "helper_sha256": h.sha256(run / HELPER),
                   "runner_sha256": h.sha256(run / Path(__file__).name),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ["torch", "triton", "flashinfer-python", "transformers", "safetensors", "apache-tvm-ffi"]},
                   "git_commit": h.command(["git", "rev-parse", "HEAD"], project),
                   "git_tracked_status": h.command(["git", "status", "--porcelain", "--untracked-files=no"], project),
                   "authorship": "coach implemented new adapter and diagnostic; no independent learner credit"}
        validate_request(request)
        h.save_json(run / "request.json", request)
        result["comparisons"] = []
        for arm in ARMS:
            out = run / arm
            out.mkdir()
            with (out / "stdout.log").open("xb") as log:
                process = subprocess.Popen([sys.executable, "-B", "-u", str(run / Path(__file__).name),
                                            "--model-path", model["path"], "--worker-run", str(run), "--arm", arm],
                                           cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in iter(process.stdout.readline, b""):
                    log.write(line)
                    log.flush()
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                code = process.wait()
            if code:
                raise RuntimeError(f"{arm} worker failed ({code}); first error/stdout preserved")
            if arm == "original":
                result["comparisons"].append(compare_arms(run / "original_repeat", out))
                h.save_json(run / "original-repeat-check.json", result["comparisons"][-1])
            elif arm == "triton":
                result["comparisons"].append(compare_arms(run / "original", out))
        baseline = json.loads((run / "original/result.json").read_text(encoding="utf-8"))
        result.update(status="MODEL_AB_CORRECTNESS_PASS", raw_expected_token_ids=baseline["raw_generated_token_ids"],
                      total_store_checks=2688, fresh_workers=3,
                      limitations="One fixed real-model workload. Diagnostic byte equality, not quality evaluation or a performance measurement.")
    except BaseException:
        primary = traceback.format_exc()
        result["error"] = primary
        (run / "controller-error.txt").write_text(primary, encoding="utf-8")
        raise
    finally:
        h.save_json(run / "result.json", result)
        try:
            args.shared_results.mkdir(parents=True, exist_ok=True)
            target = args.shared_results / run.name
            shutil.copytree(run, target)
            print("SHARED_RUN_DIRECTORY=" + str(target), flush=True)
        except Exception:
            print("SHARED_COPY_FAILED; local result preserved: " + str(run), flush=True)
            if primary is None:
                raise
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
