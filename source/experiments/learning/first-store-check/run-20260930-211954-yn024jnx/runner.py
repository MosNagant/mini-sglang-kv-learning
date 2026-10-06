"""Coach scaffolding: preserve evidence and run a learner-written check.

This is one tiny real-store case, not a benchmark or a complete test suite.
The learner writes the correctness assertion; no optimization is implemented.
"""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def main():
    project = Path.home() / "projects" / "mini-sglang-kv"
    learner = project / "experiments" / "check_real_store.py"
    if sys.argv[1:] == ["--prepare"]:
        if not project.is_dir():
            raise SystemExit(f"PROJECT_NOT_FOUND: {project}")
        learner.parent.mkdir(parents=True, exist_ok=True)
        template = Path(__file__).with_name("09-check-real-store-template.py")
        try:
            with learner.open("x", encoding="utf-8", newline="\n") as destination:
                destination.write(template.read_text(encoding="utf-8"))
        except FileExistsError:
            print(f"EXISTING_FILE_KEPT={learner}")
        else:
            print(f"LEARNER_FILE_CREATED={learner}")
        print("PREPARE_COMPLETE: edit check_first_k; no GPU test was run.")
        return
    if sys.argv[1:]:
        raise SystemExit("Usage: 09-check-real-store.py [--prepare]")
    if not learner.is_file():
        raise SystemExit("LEARNER_FILE_MISSING: run the shell script with --prepare first")

    expected_revision = "9a91cfafe754aa85daee49998176275667eb58f2"

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(project), *arguments], text=True)

    revision = git("rev-parse", "HEAD").strip()
    if revision != expected_revision:
        raise SystemExit(f"COMMIT_MISMATCH: {revision}; no checkout was changed")

    parent = project / "experiments" / "learning" / "first-store-check"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    print(f"RUN_DIRECTORY={run}", flush=True)
    print("SCOPE=learner-written first K row check; not a full suite or benchmark", flush=True)
    learner_snapshot = run / "check_real_store.py"
    shutil.copyfile(learner, learner_snapshot)
    shutil.copyfile(__file__, run / "runner.py")
    sources = [
        "python/minisgl/kernel/store.py",
        "python/minisgl/kernel/csrc/jit/store.cu",
        "python/minisgl/kernel/utils.py",
    ]
    sources.extend(git("ls-files", "--", "*warp.cuh").splitlines())
    hashes = {}
    for relative in sources:
        data = (project / relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        saved = run / "source" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(data)
    diff = git("diff", "HEAD", "--", *sources)
    (run / "source.diff").write_text(diff, encoding="utf-8")
    if diff:
        raise SystemExit("SOURCE_EDITS_PRESENT: source.diff saved; review before calling this upstream")

    import torch
    from minisgl.kernel import store_cache

    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE: preserve the error; no packages were changed")
    learner_module = runpy.run_path(str(learner_snapshot))
    checker = learner_module.get("check_first_k")
    if not callable(checker):
        raise SystemExit("CHECK_FUNCTION_MISSING: keep def check_first_k(k_cache, k)")

    # Same tiny layout and deterministic inputs as observation 08.
    kv_cache = torch.full((8, 2, 128), -1, dtype=torch.float16, device="cuda")
    k_cache, v_cache = kv_cache[:, 0, :], kv_cache[:, 1, :]
    k = torch.arange(256, dtype=torch.float16, device="cuda").reshape(2, 128) + 10
    v = k + 1000
    indices = torch.tensor([5, 1], dtype=torch.int32, device="cuda")
    store_cache(k_cache, v_cache, indices, k, v)
    torch.cuda.synchronize()

    # Save all values before executing the learner's check, even if it fails.
    record = {
        "scope": "one original CUDA store invocation; learner checks first K row only",
        "created_at": datetime.now().astimezone().isoformat(),
        "project_revision": revision,
        "source_sha256": hashes,
        "learner_sha256": hashlib.sha256(learner_snapshot.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(k.dtype),
        "k_input_stride_elements": list(k.stride()),
        "k_cache_stride_elements": list(k_cache.stride()),
        "cache_initial_value": -1,
        "indices": indices.cpu().tolist(),
        "k_input": k.cpu().tolist(),
        "v_input": v.cpu().tolist(),
        "k_cache_after": k_cache.cpu().tolist(),
        "v_cache_after": v_cache.cpu().tolist(),
    }
    (run / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"GPU={record['gpu']}", flush=True)
    print(f"RAW_RESULT_SAVED={run / 'result.json'}", flush=True)
    print("CALLING_LEARNER_CHECK=check_first_k(k_cache, k)", flush=True)
    checker(k_cache, k)
    print("LEARNER_CHECK_RETURNED: no exception; submit your assertion and this output for review.")
    print("COVERAGE_LIMIT: other K row, V, untouched rows and boundary cases are not verified here.")


if __name__ == "__main__":
    main()
