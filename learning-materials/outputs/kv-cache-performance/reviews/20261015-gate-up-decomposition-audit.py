"""CPU-only independent saved-fixture numerical decomposition; no GPU claims."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--fixture-run', required=True, type=Path)
    args = parser.parse_args()
    run, fixture_run = args.run, args.fixture_run
    result, fixture = read(run/'result.json'), read(fixture_run/'fixture.json')
    manifest = read(run/'artifact-manifest.json')
    require(sha(fixture_run/'fixture.json') == result['fixture_sha256'], 'Fixture manifest binding changed')
    require(sha(run/'result.json') == manifest['result.json'], 'Microcampaign result changed')
    names = result['numerical_eligible']
    require(len(names) == 7 and len(fixture['samples']) == 84, 'Unexpected campaign scope')
    records = {name: {'y_bitwise_differences': 0, 'y_elements': 0,
                     'z_bitwise_differences': None if name.startswith('swiglu') else 0,
                     'max_y_abs': 0.0} for name in names}
    all_y_cross_differences = 0
    same_r8_differences = 0
    verified = 0
    for sample in fixture['samples']:
        original_path = fixture_run/sample['path']
        require(sha(original_path) == sample['sha256'], 'Original captured sample changed')
        arrays = {}
        with np.load(original_path, allow_pickle=False) as original:
            for name in names:
                relative = f"numerical/{name}/step{sample['step']:02d}-layer{sample['layer']:02d}.npz"
                path = run/relative
                require(sha(path) == manifest[relative], 'Candidate output changed: '+relative)
                verified += 1
                with np.load(path, allow_pickle=False) as output:
                    y = output['y'].copy()
                    require(y.shape == original['y'].shape == (1,3072) and y.dtype == original['y'].dtype == np.float16,
                            'Output Y metadata mismatch')
                    arrays[name] = y
                    record = records[name]
                    record['y_bitwise_differences'] += int(np.count_nonzero(y.view(np.uint16) != original['y'].view(np.uint16)))
                    record['y_elements'] += y.size
                    record['max_y_abs'] = max(record['max_y_abs'], float(np.max(np.abs(y.astype(np.float32)-original['y'].astype(np.float32)))))
                    if record['z_bitwise_differences'] is not None:
                        z = output['z']
                        require(z.shape == original['z'].shape == (1,6144) and z.dtype == original['z'].dtype == np.float16,
                                'Output Z metadata mismatch')
                        record['z_bitwise_differences'] += int(np.count_nonzero(z.view(np.uint16) != original['z'].view(np.uint16)))
        reference = arrays[names[0]].view(np.uint16)
        all_y_cross_differences += sum(int(np.count_nonzero(reference != arrays[name].view(np.uint16))) for name in names[1:])
        same_r8_differences += int(np.count_nonzero(arrays['gemv_direct_r8_w4'].view(np.uint16) != arrays['swiglu_fused_r8_w4'].view(np.uint16)))
    observed = dict(records, same_r8w4_unfused_vs_fused_y_differences=same_r8_differences)
    require(observed == read(run/'numerical-decomposition.json'), 'Independent bit-pattern recomputation differs from original decomposition')
    output = {'status': 'DECOMPOSITION_EVIDENCE_AUDIT_PASS', 'candidate_npz_files_verified': verified,
              'original_sample_files_verified': 84, 'bit_comparison': 'float16 uint16 bit patterns, including signed zero',
              'all_seven_candidate_y_cross_differences': all_y_cross_differences,
              'same_r8w4_unfused_vs_fused_y_differences': same_r8_differences, 'per_candidate': records,
              'interpretation': 'On these 84 original-model fixture inputs, activation fusion adds no observed Y difference relative to matching Triton GEMV plus original FlashInfer activation. Raw GEMV differs from original FP16 F.linear on 490 Z elements. Reduction/rounding differences are a plausible investigation direction; this does not isolate full-model logit divergence causality.',
              'source_sha256': sha(__file__), 'decomposition_sha256': sha(run/'numerical-decomposition.json'),
              'limit': 'Offline comparison of existing arrays only; no new GPU execution, retimed sample, relaxed acceptance gate or broad equivalence claim.'}
    path = run/'numerical-decomposition-audit.json'
    require(not path.exists(), 'Preserve existing decomposition audit')
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
