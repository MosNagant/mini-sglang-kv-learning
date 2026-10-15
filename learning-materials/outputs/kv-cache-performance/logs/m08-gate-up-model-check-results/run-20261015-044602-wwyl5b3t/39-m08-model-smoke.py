"""M08: real local Qwen3-0.6B, original KV store, diagnostic eager trace.

No downloads, dummy weights, timing, kernel replacement, or upstream file edits.
The controller preserves a fresh source snapshot, worker stdout, and first failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
ORIGINAL_PINS = {
    "python/minisgl/kernel/store.py": "39e21e21d1de0cc2f14faa37384ea94e427c0aabc997a56c049884ed46fc0c01",
    "python/minisgl/kvcache/mha_pool.py": "c22ce82ac132be5dc18a26b8fbb58239cff9baa6f1053aee896376059cb9c4fe",
    "python/minisgl/kernel/csrc/jit/store.cu": "8207607fb922e63236ee5c24f09e2cf45088c045d05ca8a92f6215af04f8e25e",
    "python/minisgl/kernel/csrc/include/minisgl/warp.cuh": "ebd7e93012be03ebfa8cdafc3ec1836ac32e17ad29a060991ed604dd25d4a410",
}
ENGINE_OPTIONS = {
    "max_running_req": 1, "attention_backend": "fi", "page_size": 1,
    "num_page_override": 256, "max_seq_len_override": 128,
    "max_extend_tokens": 128, "cuda_graph_bs": [], "memory_ratio": 0.3,
    "use_dummy_weight": False, "use_pynccl": False, "cache_type": "naive",
}
PACKAGES = ["torch", "triton", "transformers", "safetensors", "flashinfer-python",
            "apache-tvm-ffi", "sgl-kernel", "quack-kernels", "huggingface-hub"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command(args, cwd):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
    return {"args": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def inspect_model(model_path, revision=None):
    """Inspect local checkpoint bytes, without importing Torch or Transformers."""
    model_path = Path(model_path).expanduser().resolve(strict=True)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in config.get("architectures", []):
        raise ValueError("This lesson requires a local Qwen3ForCausalLM checkpoint")
    for name, value in {"hidden_size": 1024, "num_hidden_layers": 28, "head_dim": 128,
                        "num_attention_heads": 16, "num_key_value_heads": 8}.items():
        if config.get(name) != value:
            raise ValueError(f"Qwen3-0.6B configuration mismatch: {name}")
    if config.get("quantization_config"):
        raise ValueError("Quantized checkpoints are outside this FP16 smoke scope")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (model_path / name).is_file():
            raise ValueError(f"Local tokenizer is incomplete: {name}")
    index = model_path / "model.safetensors.index.json"
    files = sorted(p for p in model_path.glob("*.safetensors") if p.name != "consolidated.safetensors")
    if not files:
        raise ValueError("Real local safetensors weights are required; no dummy fallback")
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        names = set(weight_map.values())
        if not names or any(Path(n).name != n or not n.endswith(".safetensors") for n in names):
            raise ValueError("Invalid safetensors index paths")
        if names != {p.name for p in files}:
            raise ValueError("Missing or unexpected local checkpoint shards")
    weights = []
    for path in files:
        with path.open("rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                raise ValueError(f"Truncated safetensors header: {path.name}")
            length = struct.unpack("<Q", head)[0]
            if not 1 <= length <= min(64 << 20, path.stat().st_size - 8):
                raise ValueError(f"Invalid safetensors header length: {path.name}")
            tensors = json.loads(handle.read(length))
        tensors.pop("__metadata__", None)
        if not tensors:
            raise ValueError(f"Empty weights: {path.name}")
        data_bytes = path.stat().st_size - 8 - length
        for item in tensors.values():
            offsets = item.get("data_offsets", [])
            if len(offsets) != 2 or not 0 <= offsets[0] < offsets[1] <= data_bytes:
                raise ValueError(f"Truncated/invalid tensor storage: {path.name}")
        stat = path.stat()
        weights.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "sha256": sha256(path), "tensor_count": len(tensors)})
    detected = model_path.name if model_path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", model_path.name) else None
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--model-revision must be a full 40-character commit hash")
    if revision and detected and revision != detected:
        raise ValueError("Declared model revision disagrees with snapshot directory")
    metadata = {p.name: sha256(p) for p in sorted(model_path.iterdir())
                if p.is_file() and p.suffix in (".json", ".txt", ".jinja")}
    return {"path": str(model_path), "requested_model": "Qwen/Qwen3-0.6B", "config": config,
            "revision": revision or detected, "revision_basis": "declared by caller" if revision else
            ("Hugging Face snapshot directory name" if detected else "unavailable; use file hashes"),
            "weights": weights, "metadata_sha256": metadata,
            "weight_policy": "Local safetensors loaded by original loader; use_dummy_weight=False. Hashes identify bytes, not publisher authenticity."}


def snapshot_source(project, run):
    source = run / "source"
    records = {}
    for relative, digest in ORIGINAL_PINS.items():
        if sha256(project / relative) != digest:
            raise ValueError(f"Original KV source changed: {relative}")
    paths = sorted(p for p in (project / "python/minisgl").rglob("*")
                   if p.is_file() and p.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp"))
    paths += [project / "pyproject.toml", project / "benchmark/offline/bench.py"]
    for path in paths:
        relative = path.relative_to(project).as_posix()
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records[relative] = sha256(target)
        if records[relative] != sha256(path):
            raise RuntimeError(f"Source changed while snapshotting: {relative}")
    shutil.copyfile(__file__, run / "runner.py")
    shell = HERE / "39-m08-model-smoke.sh"
    if shell.is_file():
        shutil.copyfile(shell, run / shell.name)
    save_json(run / "source-manifest.json", records)
    return records


def tensor_metadata(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "storage_offset": tensor.storage_offset(),
            "element_size": tensor.element_size(), "contiguous": tensor.is_contiguous(),
            "data_ptr": tensor.data_ptr(), "storage_ptr": tensor.untyped_storage().data_ptr(),
            "storage_nbytes": tensor.untyped_storage().nbytes()}


class KVTrace:
    """Observe the real pool -> original store call; never substitute a kernel."""
    def __init__(self, path, original_store, original_pool_store, get_context):
        self.handle = Path(path).open("x", encoding="utf-8")
        self.original_store = original_store
        self.original_pool_store = original_pool_store
        self.get_context = get_context
        self.active_layer = None
        self.records = []

    def emit(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def pool_store(self, pool, k, v, out_loc, layer_id):
        if self.active_layer is not None:
            raise RuntimeError("Nested KV pool call is outside this diagnostic scope")
        self.active_layer = int(layer_id)
        try:
            return self.original_pool_store(pool, k, v, out_loc, layer_id)
        finally:
            self.active_layer = None

    def store(self, k_cache, v_cache, indices, k, v):
        if self.active_layer is None:
            raise RuntimeError("Original store called without layer attribution")
        batch = self.get_context().batch
        record = {"call_id": len(self.records), "layer": self.active_layer,
                  "phase": batch.phase, "requests": batch.size, "padded_requests": batch.padded_size,
                  "tokens_B": int(k.shape[0]), "flattened_D": int(k.numel() // k.shape[0]),
                  "input_token_ids": batch.input_ids.detach().cpu().tolist(),
                  "positions": batch.positions.detach().cpu().tolist(),
                  "indices": indices.detach().cpu().tolist(),
                  "tensors": {name: tensor_metadata(t) for name, t in
                              [("k_cache", k_cache), ("v_cache", v_cache), ("indices", indices), ("k", k), ("v", v)]},
                  "status": "BEFORE_ORIGINAL_CALL"}
        self.records.append(record)
        self.emit(record)
        try:
            result = self.original_store(k_cache, v_cache, indices, k, v)
        except BaseException:
            record["status"] = "ORIGINAL_CALL_FAILED"
            record["error"] = traceback.format_exc()
            try:
                self.emit(record)
            except Exception:
                pass  # Preserve the original store exception.
            raise
        record["status"] = "ORIGINAL_CALL_RETURNED"
        self.emit(record)
        return result


def validate_trace(records, layers, output_count):
    expected = Counter({(phase, layer): count for layer in range(layers)
                        for phase, count in [("prefill", 1), ("decode", output_count - 1)]})
    actual = Counter((r["phase"], r["layer"]) for r in records)
    if actual != expected or any(r["status"] != "ORIGINAL_CALL_RETURNED" for r in records):
        raise RuntimeError(f"Incomplete real prefill/decode store trace: {actual}")
    for r in records:
        if r["requests"] != 1 or r["tokens_B"] != len(r["indices"]):
            raise RuntimeError("Unexpected diagnostic batch shape")
        if r["phase"] == "decode" and r["tokens_B"] != 1:
            raise RuntimeError("Decode trace must contain one real token row")
        if len(set(r["indices"])) != len(r["indices"]):
            raise RuntimeError("Unexpected duplicate cache write indices")
        if any(i < 0 or i >= r["tensors"]["k_cache"]["shape"][0] for i in r["indices"]):
            raise RuntimeError("Trace indices outside actual cache")
    return {phase: sum(r["phase"] == phase for r in records) for phase in ("prefill", "decode")}


def worker(run):
    request = json.loads((run / "request.json").read_text(encoding="utf-8"))
    source = run / "source"
    hashes = json.loads((run / "source-manifest.json").read_text(encoding="utf-8"))
    for relative, digest in hashes.items():
        if sha256(source / relative) != digest:
            raise ValueError(f"Saved source hash mismatch: {relative}")
    sys.path.insert(0, str(source / "python"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", MINISGL_DISABLE_OVERLAP_SCHEDULING="1")
    import torch
    import minisgl.kernel as kernel_api
    import minisgl.kernel.store as store_module
    from minisgl.core import SamplingParams, get_global_ctx
    from minisgl.kvcache.mha_pool import MHAKVCache
    from minisgl.llm import LLM
    from minisgl.utils import load_tokenizer

    # No CUDA query, tensor, or device selection before LLM -> Engine initialization.
    original_store, original_pool = store_module.store_cache, MHAKVCache.store_kv
    if kernel_api.store_cache is not original_store:
        raise RuntimeError("Original public store alias is already replaced")
    if Path(inspect.getsourcefile(original_store)).resolve() != source / "python/minisgl/kernel/store.py":
        raise RuntimeError("Wrong minisgl import path")
    trace = KVTrace(run / "kv-trace.jsonl", original_store, original_pool, get_global_ctx)
    raw_tokens = []

    class ObservedLLM(LLM):
        def offline_send_result(self, reply):
            for msg in reply:
                raw_tokens.append({"uid": msg.uid, "token_id": int(msg.next_token), "finished": msg.finished})
            save_json(run / "raw-output-tokens.json", raw_tokens)
            return super().offline_send_result(reply)

    llm = None
    primary = None
    result = {"status": "FAILED", "scope": "real model original-KV diagnostic; no timing or optimized arm",
              "cuda_graph": False, "overlap_scheduling": False, "tp": 1, "dtype": "torch.float16",
              "original_store_file": str(inspect.getsourcefile(original_store))}
    try:
        tokenizer = load_tokenizer(request["model"]["path"])
        input_ids = tokenizer.encode(request["prompt"])
        if not 2 <= len(input_ids) <= 96 or len(input_ids) + request["max_new_tokens"] > 128:
            raise ValueError("Smoke prompt must contain 2..96 tokens and fit the 128-token sequence")
        save_json(run / "input.json", {"prompt": request["prompt"], "input_token_ids": input_ids})
        kernel_api.store_cache = trace.store
        MHAKVCache.store_kv = lambda pool, k, v, out_loc, layer_id: trace.pool_store(pool, k, v, out_loc, layer_id)
        llm = ObservedLLM(request["model"]["path"], dtype=torch.float16, **ENGINE_OPTIONS)
        if llm.engine.graph_runner.max_graph_bs != 0:
            raise RuntimeError("Diagnostic requires real eager callbacks, CUDA Graph must be disabled")
        state = llm.engine.model.state_dict()
        if not state or any(t.device.type != "cuda" or t.dtype != torch.float16 for t in state.values()):
            raise RuntimeError("Model weights were not materialized as CUDA FP16 tensors")
        result["materialized_model"] = {"tensor_count": len(state), "parameters": sum(t.numel() for t in state.values())}
        result["environment"] = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                                 "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                                 "capability": list(torch.cuda.get_device_capability(0)),
                                 "free_total_bytes": list(torch.cuda.mem_get_info()),
                                 "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], run)}
        result["cache_allocation"] = tensor_metadata(llm.engine.kv_cache._kv_buffer)
        result["usable_cache_tokens"] = llm.engine.num_pages
        result["dummy_cache_page"] = 1
        save_json(run / "initialization.json", result)
        outputs = llm.generate([input_ids], SamplingParams(temperature=0.0, ignore_eos=True,
                                                          max_tokens=request["max_new_tokens"]))
        torch.cuda.synchronize()
        save_json(run / "outputs.json", outputs)
        if len(raw_tokens) != request["max_new_tokens"] or any(t["uid"] != 0 for t in raw_tokens):
            raise RuntimeError("Generation did not produce the requested real token sequence")
        if any(not 0 <= t["token_id"] < request["model"]["config"]["vocab_size"] for t in raw_tokens):
            raise RuntimeError("Invalid generated token ID")
        result["phase_store_counts"] = validate_trace(trace.records, request["model"]["config"]["num_hidden_layers"], len(raw_tokens))
        result["store_calls"] = len(trace.records)
        result["raw_generated_token_ids"] = [t["token_id"] for t in raw_tokens]
        result["outputs"] = outputs
        for item in request["model"]["weights"]:
            stat = (Path(request["model"]["path"]) / item["name"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise RuntimeError("Model file changed during loading/generation")
        result["status"] = "MODEL_SMOKE_AND_KV_TRACE_COMPLETE"
        result["interpretation"] = "Real weights and generated tokens; all layers observed in prefill/decode, original calls returned and CUDA synchronized. This is not an independent model-quality/numerical-reference test or a performance measurement."
    except BaseException:
        primary = traceback.format_exc()
        result["error"] = primary
        (run / "error.txt").write_text(primary, encoding="utf-8")
        raise
    finally:
        kernel_api.store_cache = original_store
        MHAKVCache.store_kv = original_pool
        try:
            trace.handle.close()
            if llm is not None:
                llm.shutdown()
        except BaseException:
            result["shutdown_error"] = traceback.format_exc()
            if primary is None:
                result["status"] = "FAILED"
                save_json(run / "result.json", result)
                raise
        save_json(run / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home() / "projects/mini-sglang-kv")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-revision", help="Optional full upstream model commit; recorded as caller declaration")
    parser.add_argument("--shared-results", type=Path, default=HERE / "logs/m08-model-smoke-results")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, choices=range(2, 17), default=8)
    parser.add_argument("--dry-run", action="store_true", help="CPU-only source/model-file inspection; no Torch import")
    parser.add_argument("--worker-run", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_run:
        worker(args.worker_run.resolve())
        return
    project = args.project.expanduser().resolve(strict=True)
    if args.dry_run:
        for relative, digest in ORIGINAL_PINS.items():
            if sha256(project / relative) != digest:
                raise ValueError(f"Original source changed: {relative}")
        print(json.dumps({"status": "CPU_INSPECTION_ONLY", "model": inspect_model(args.model_path, args.model_revision),
                          "engine_options": ENGINE_OPTIONS, "dtype": "FP16", "tp": 1,
                          "max_new_tokens": args.max_new_tokens, "diagnostic_not_timing": True}, ensure_ascii=False, indent=2))
        return
    local_results = project / "experiments/m08-model-smoke-results"
    local_results.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") , dir=local_results))
    print("RUN_DIRECTORY=" + str(run), flush=True)
    primary = None
    try:
        source_hashes = snapshot_source(project, run)
        packages = {}
        for name in PACKAGES:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        request = {"model": inspect_model(args.model_path, args.model_revision), "prompt": args.prompt,
                   "max_new_tokens": args.max_new_tokens, "engine_options": ENGINE_OPTIONS,
                   "packages": packages, "source_hashes": source_hashes, "runner_sha256": sha256(__file__),
                   "git_commit": command(["git", "rev-parse", "HEAD"], project),
                   "git_tracked_status": command(["git", "status", "--porcelain", "--untracked-files=no"], project),
                   "authorship": "Coach authored runner and diagnostic instrumentation; original upstream model/KV implementation executes unchanged."}
        save_json(run / "request.json", request)
        with (run / "stdout.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(run / "runner.py"),
                                        "--model-path", request["model"]["path"], "--worker-run", str(run)],
                                       cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(process.stdout.readline, b""):
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f"Model smoke worker failed with exit code {code}; see preserved stdout/error files")
        if json.loads((run / "result.json").read_text(encoding="utf-8"))["status"] != "MODEL_SMOKE_AND_KV_TRACE_COMPLETE":
            raise RuntimeError("Worker did not complete the model smoke and trace")
    except BaseException:
        primary = traceback.format_exc()
        (run / "controller-error.txt").write_text(primary, encoding="utf-8")
        raise
    finally:
        try:
            args.shared_results.mkdir(parents=True, exist_ok=True)
            target = args.shared_results / run.name
            shutil.copytree(run, target)
            print("SHARED_RUN_DIRECTORY=" + str(target), flush=True)
        except Exception:
            print("SHARED_COPY_FAILED; local result preserved: " + str(run), flush=True)
            if primary is None:
                raise


if __name__ == "__main__":
    main()
