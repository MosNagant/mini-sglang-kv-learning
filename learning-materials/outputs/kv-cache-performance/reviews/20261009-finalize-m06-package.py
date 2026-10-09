"""Seal M06 files, verify Markdown links and ZIP bytes, using no GPU."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import zipfile

base = Path(__file__).resolve().parents[1]
root = base / 'modules' / 'M06'
files = {}
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'MANIFEST.json':
        data = path.read_bytes()
        files[path.relative_to(root).as_posix()] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
manifest = dict(frozen='2026-10-09', scope='M06 accepted source and evidence; no fresh GPU execution',
                exclusions=['reproducible __pycache__ directories', 'this manifest itself'], files=files)
(root / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

checked_links = 0
for path in [*root.glob('*.md'), base/'lessons/m07-01-num-warps-hypothesis.md', base/'modules/README.md']:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        if '://' in target or target.startswith('#'):
            continue
        target = target.split('#')[0]
        assert (path.parent / target).exists(), (path.name, target)
        checked_links += 1
subprocess.run([sys.executable, str(root/'verify_package.py')], check=True)

archive = base / 'modules' / 'M06-v1-20261009.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(root.rglob('*')):
        if path.is_file():
            z.write(path, (Path('M06') / path.relative_to(root)).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(files)+1
    for rel, entry in files.items():
        assert hashlib.sha256(z.read('M06/'+rel)).hexdigest() == entry['sha256']

receipt = dict(archive=str(archive), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               archive_bytes=archive.stat().st_size, payload_files=len(files),
               payload_bytes=sum(x['bytes'] for x in files.values()), checked_markdown_links=checked_links,
               scope='local copy, file hashes, arithmetic, links and ZIP integrity; no new GPU run',
               wsl_installed=False, github_uploaded=False)
(base/'modules/M06-package-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
