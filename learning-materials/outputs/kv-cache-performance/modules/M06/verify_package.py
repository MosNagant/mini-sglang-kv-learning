"""Verify frozen package bytes using only the Python standard library."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
for name, entry in manifest['files'].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SystemExit(f'MISSING_OR_UNSAFE: {name}')
    data = path.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise SystemExit(f'CHANGED: {name}')
print(f"PACKAGE_HASHES_OK: {len(manifest['files'])} files; no GPU run performed")
