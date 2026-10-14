"""Archive selected, hash-verified source files from the real October14 WSL export.

Never changes the WSL project or old evidence. Run from any directory with Python.
"""
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPLOAD = HERE.parents[2] / "github-upload"
ARCHIVE = UPLOAD / "wsl-project-20261014T105320Z.zip"
EXPECTED_ARCHIVE = "a973db75652b953e5fba7f274d393a8802058efbaead323b11f6045863c87093"
FILES = (
    "experiments/triton_store_mask.py",
    "experiments/triton_store_api.py",
    "experiments/triton_store_candidate.py",
    "experiments/triton_store_grouped.py",
    "experiments/triton_store_full_groups.py",
    "experiments/kv_contract_support.py",
    "experiments/check_kv_targets.py",
    "python/minisgl/kernel/store.py",
    "python/minisgl/kernel/csrc/jit/store.cu",
    "LICENSE",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    record = json.loads((HERE / "source-manifest.json").read_text(encoding="utf-8"))
    for item in record["files"]:
        data = (HERE / item["path"]).read_bytes()
        if sha(data) != item["sha256"] or len(data) != item["bytes"]:
            raise ValueError("ARCHIVED_SOURCE_CHANGED: " + item["path"])
    print("SOURCE_PACKAGE_VERIFIED:", len(record["files"]), "files; no GPU run")


def main():
    data = ARCHIVE.read_bytes()
    if sha(data) != EXPECTED_ARCHIVE:
        raise ValueError("WSL_EXPORT_CHANGED")
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        exported = json.loads(archive.read("EXPORT_MANIFEST.json"))
        manifest = {item["path"]: item for item in exported["files"]}
        if exported["commit"] != "9a91cfafe754aa85daee49998176275667eb58f2":
            raise ValueError("UPSTREAM_CHANGED")
        for name in FILES:
            content = archive.read(name)
            if sha(content) != manifest[name]["sha256"]:
                raise ValueError("EXPORT_ENTRY_CHANGED: " + name)
            relative = ("src/" + Path(name).name) if name.startswith("experiments/") else "original_source/" + name
            target = HERE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            records.append(dict(path=relative, exported_path=name, bytes=len(content), sha256=sha(content)))
    result = dict(
        date="2026-10-14", source_export=ARCHIVE.name, source_export_sha256=EXPECTED_ARCHIVE,
        upstream_commit=exported["commit"], files=records,
        scope="Selected actual WSL source snapshot, not a standalone install or new GPU result; helpers/runners and raw data remain in the parent lab.",
        ownership="Historical learner/coaching code preserved byte-for-byte; October14 packaging by coach under explicit authorization.",
    )
    (HERE / "source-manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify()


if __name__ == "__main__":
    import sys
    verify() if "--verify" in sys.argv else main()
