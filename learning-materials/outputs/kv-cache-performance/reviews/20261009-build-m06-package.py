"""Copy immutable M06 evidence into a review block; no WSL/GPU execution."""
from pathlib import Path
import hashlib
import json
import shutil
import statistics

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'modules' / 'M06'
RUNS = {
    '01-width': 'kv-width-check-results/run-20261007-154657-9g3dxths',
    '02-layout': 'kv-width-check-results/run-20261007-164300-ejjr_j5x',
    '03-mask': 'm06-mask-results/run-20261007-181154-5q2k4kbh',
    '04-wrapper': 'm06-wrapper-results/run-20261009-134214-o635nwfm',
    '05-common': 'm06-common-results/run-20261009-140928-ht8v9mg8',
    '07-compile': 'm06-compile-results/run-20261009-142759-l12p0a0c',
    '06-paired': 'm06-paired-timing-results/run-20261009-145144-n1lwky52',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha(source) != sha(dest):
            raise RuntimeError(f'Preserve existing different file: {dest}')
    else:
        shutil.copyfile(source, dest)
    assert sha(source) == sha(dest)

index = {}
for label, rel in RUNS.items():
    source = BASE / 'logs' / rel
    assert source.is_dir(), source
    target = OUT / 'evidence' / label
    count = 0
    for file in sorted(source.rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts:
            copy(file, target / file.relative_to(source))
            count += 1
    hashes = source / 'source_hashes.json'
    if hashes.exists():
        for name, expected in json.loads(hashes.read_text(encoding='utf-8')).items():
            saved = source / name
            if not saved.is_file() and name.startswith('python/'):
                saved = source / 'original_source' / name
            assert sha(saved) == expected, (label, name)
    index[label] = dict(original=str(source), package=f'evidence/{label}', files=count)

paired = OUT / 'evidence' / '06-paired'
for name in ('triton_store_mask.py', 'triton_store_api.py', 'kv_contract_support.py', 'check_kv_targets.py'):
    copy(paired / name, OUT / 'src' / name)
copy(paired / 'learner-timing.py', OUT / 'src' / 'learner-timing.py')

runners = ('14-check-kv-widths.py', '15-check-kv-contract.py', '21-check-kv-mask.py',
           '21-triton-store-mask.py', '22-check-kv-wrapper.py', '22-check-kv-wrapper.sh',
           '22-kv-contract-support.py', '22-triton-store-api-template.py',
           '23-check-kv-common.py', '23-check-kv-common.sh',
           '24-inspect-triton-compile.py', '24-inspect-triton-compile.sh',
           '25-compare-kv-v1.py', '25-compare-kv-v1.sh')
for name in runners:
    options = [paired / name, OUT / 'evidence' / '07-compile' / name, BASE / name]
    source = next(p for p in options if p.is_file())
    copy(source, OUT / 'runners' / name)

for name in ('kv-wrapper-check-20261009-134213-Q0P2TM.log',
             'kv-common-check-20261009-140928-ACOM20.log',
             'kv-compile-inspect-20261009-142759-8rqgRU.log',
             'kv-paired-timing-20261009-145144-K17PKm.log'):
    copy(BASE / 'logs' / name, OUT / 'evidence' / 'console' / name)

summary = json.loads((paired / 'summary.json').read_text(encoding='utf-8'))
rows = summary['summary']
audit = []
for row in rows:
    b, d = row['batch'], row['width']
    item = dict(batch=b, width=d)
    for policy in ('graph', 'eager'):
        medians = {}
        for impl in ('original_cuda', 'learner_triton'):
            medians[impl] = []
            for r in (1, 2, 3):
                result = json.loads((paired / f'round-{r}-B{b}-D{d}' / 'result.json').read_text(encoding='utf-8'))
                samples = result['samples_ms_per_call'][impl][policy]
                assert len(samples) == 30 and all(x > 0 for x in samples)
                medians[impl].append(statistics.median(samples) * 1000)
            assert medians[impl] == row[policy][impl]['round_medians_us']
            assert statistics.median(medians[impl]) == row[policy][impl]['representative_us']
        orig = statistics.median(medians['original_cuda'])
        triton = statistics.median(medians['learner_triton'])
        ratios = [o/t for o,t in zip(medians['original_cuda'], medians['learner_triton'])]
        assert ratios == row[policy]['paired_round_speedups']
        assert orig / triton == row[policy]['speedup_original_over_triton']
        item[policy] = dict(original_us=orig, triton_us=triton, ratio=orig/triton,
                            time_reduction_pct=100*(1-triton/orig), paired_ratios=ratios,
                            winning_rounds=sum(r > 1 for r in ratios))
    audit.append(item)

(OUT / 'evidence-index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(OUT / 'statistics-review.json').write_text(json.dumps(dict(scope='offline arithmetic and copied-file hash checks; no new GPU run', cases=audit), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(package=str(OUT), evidence=index, statistics=audit), ensure_ascii=False, indent=2))
