"""Archive installed FlashInfer activation sources without importing FlashInfer/Torch.

This records available source and installation metadata, not proof of the runtime
kernel's compiled flags, architecture specialization, or numerical equivalence.
Only stdlib imports are used. No package installation, network or CUDA calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TARGET_SUFFIXES = (
    "flashinfer/activation.py", "flashinfer/jit/activation.py",
    "include/flashinfer/activation.cuh", "include/flashinfer/math.cuh",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output must be new or empty; preserving existing evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)
    modules_before = set(sys.modules)
    # Top-level find_spec does not execute the package. Do not find_spec a dotted
    # child here: that could import its parent and initialize GPU-dependent code.
    spec = importlib.util.find_spec("flashinfer")
    if spec is None:
        raise RuntimeError("Installed flashinfer package was not found on this Python path")
    roots = [Path(p).resolve() for p in (spec.submodule_search_locations or [])]
    paths = set()
    for root in roots:
        for relative in ("activation.py", "jit/activation.py", "include/flashinfer/activation.cuh",
                         "include/flashinfer/math.cuh", "data/include/flashinfer/activation.cuh",
                         "data/include/flashinfer/math.cuh"):
            path = root / relative
            if path.is_file():
                paths.add(path.resolve())
    distributions = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if "flashinfer" not in name.lower().replace("_", "-"):
            continue
        item = {"name": name, "version": dist.version,
                "installation_root": str(Path(dist.locate_file("")).resolve())}
        distributions.append(item)
        for entry in dist.files or []:
            normalized = str(entry).replace("\\", "/")
            if any(normalized.endswith(suffix) for suffix in TARGET_SUFFIXES):
                path = Path(dist.locate_file(entry)).resolve()
                if path.is_file():
                    paths.add(path)
    records = []
    for index, path in enumerate(sorted(paths, key=str)):
        destination = out / "sources" / f"{index:02d}-{path.name}"
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        source = path.read_text(encoding="utf-8", errors="replace")
        excerpts = [{"line": number, "text": line.strip()} for number, line in enumerate(source.splitlines(), 1)
                    if re.search(r"silu|sigmoid|__expf|\bexpf\b|ptx_exp|ex2|fast_math|exp2|\bexp\(", line, re.I)]
        records.append({"original_path": str(path), "archived_path": str(destination.relative_to(out)),
                        "bytes": path.stat().st_size, "sha256": sha(path), "activation_math_lines": excerpts})
    observed_paths = [record["original_path"].replace("\\", "/") for record in records]
    missing = [suffix for suffix in TARGET_SUFFIXES if not any(path.endswith(suffix) for path in observed_paths)]
    newly_loaded = sorted(set(sys.modules) - modules_before)
    forbidden = [name for name in newly_loaded if name == "torch" or name.startswith(("torch.", "flashinfer."))
                 or name == "flashinfer"]
    if forbidden:
        raise RuntimeError(f"Unexpected package execution while reading installation metadata: {forbidden}")
    manifest = {
        "status": "SOURCES_ARCHIVED" if not missing else "PARTIAL_SOURCE_ARCHIVE",
        "utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
        "flashinfer_module_origin": spec.origin, "flashinfer_package_roots": list(map(str, roots)),
        "distributions": distributions, "files": records,
        "missing_required_source_suffixes": missing,
        "dotted_submodule_imports": False, "gpu_execution": False,
        "limit": "Installed source evidence only; no compiled-kernel, compiler-flags or numerical-equivalence claim.",
        "script_sha256": sha(__file__),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "files": len(records),
                      "manifest": str(out / "manifest.json"), "missing": manifest["missing_required_source_suffixes"]}))


if __name__ == "__main__":
    main()
