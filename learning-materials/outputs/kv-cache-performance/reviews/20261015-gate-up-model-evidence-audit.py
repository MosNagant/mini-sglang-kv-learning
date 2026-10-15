"""Independently audit frozen 51 strict-model data on CPU, including rejection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise AssertionError(message)


def audit(run):
    cfg, result = read(run/'plan.json'), read(run/'result.json')
    require(cfg['mode'] == result['mode'] == 'check', 'Audit currently covers strict Eager model checks')
    require(result['plan_sha256'] == sha(run/'plan.json'), 'Plan digest mismatch')
    require(result['order_sha256'] == sha(run/'order.json'), 'Order digest mismatch')
    require(read(run/'order.json') == [{'arm': arm, 'policy': 'eager', 'round': 0}
                                      for arm in ('original', 'candidate')], 'Fresh worker order changed')
    for name, digest in cfg['frozen_files'].items():
        require(sha(run/name) == digest, 'Frozen experiment file changed: '+name)
    for name, digest in cfg['source_manifest'].items():
        require(sha(run/'source'/name) == digest, 'Original source changed: '+name)
    require(sha(run/'candidate.py') == cfg['candidate_source_sha256'], 'Candidate binding mismatch')
    micro = read(run/'accepted-micro-result.json')
    fixture = read(run/'accepted-fixture.json')
    fixture_plan = read(run/'accepted-fixture-plan.json')
    require(micro['status'] == 'GATE_UP_CAMPAIGN_COMPLETE', 'Unfinished microcampaign')
    require(micro['candidate_source_sha256'] == cfg['candidate_source_sha256'], 'Changed candidate since microcampaign')
    require(micro['fixture_sha256'] == cfg['fixture_sha256'] == sha(run/'accepted-fixture.json'), 'Changed fixture')
    require(micro['selected_candidate'] == cfg['candidate_name'] and cfg['candidate_name'] in micro['numerical_eligible'], 'Candidate not the frozen eligible selection')
    require(micro['holdout']['status'] == 'PASS', 'Holdout numerical/collection gate failed')
    require(micro['selection']['strong_local_gain'] or cfg['investigator_candidate'], 'Missing weak-gain investigator declaration')
    require(fixture['status'] == 'GATE_UP_FIXTURE_CAPTURE_PASS', 'Fixture capture failed')
    require(fixture['plan_sha256'] == sha(run/'accepted-fixture-plan.json'), 'Fixture plan changed')
    for key in ('source_manifest', 'input_token_ids', 'raw_expected_token_ids', 'timing_packages', 'accepted_device'):
        require(cfg[key] == fixture_plan[key], 'Changed fixture-to-model binding: '+key)
    require(cfg['model']['metadata_sha256'] == fixture_plan['model']['metadata_sha256'], 'Changed model/tokenizer metadata hashes')
    require([(w['name'],w['sha256']) for w in cfg['model']['weights']] ==
            [(w['name'],w['sha256']) for w in fixture_plan['model']['weights']], 'Changed model weight hashes')
    workers = {}
    for arm in ('original', 'candidate'):
        worker = read(run/arm/'result.json')
        workers[arm] = worker
        require(worker in result['workers'], 'Worker/controller evidence mismatch')
        require(worker['status'] == 'MODEL_ALL_LOGITS_COLLECTED' and worker['arm'] == arm, 'Worker incomplete')
        require(worker['policy'] == 'eager' and worker['mode'] == 'check', 'Wrong strict execution policy')
        require(worker['logits_sha256'] == sha(run/arm/'all-logits.npz'), 'Saved logits hash mismatch')
        require(worker['logits_count'] == 32 and worker['input_token_ids'] == cfg['input_token_ids'], 'Wrong workload')
        require(worker['environment']['packages'] == cfg['timing_packages'], 'Different runtime packages')
        for name, value in cfg['accepted_device'].items():
            require(worker['environment'][name] == value, 'Different GPU environment')
        require(worker['numerical_contract'] == {'atol': 0.002, 'rtol': 0.002, 'every_element': True}, 'Changed tolerance')
    evidence = workers['candidate']['candidate_evidence']
    require(evidence['counters'] == {'decode_python_calls': 868, 'capture_python_calls': 0,
            'prefill_original_python_calls': 28, 'original_decode_fallback_calls': 0, 'current_input_calls': 868},
            'Actual candidate routing counts differ')
    require(len(evidence['layers']) == 28, 'Missing layer evidence')
    for index, layer in enumerate(evidence['layers']):
        require(layer['layer'] == index and layer['candidate'] == cfg['candidate_name'], 'Layer selection mismatch')
        require(layer['input_policy'] == 'call_with_input(current x)', 'Wrong current-input policy')
        require(layer['weight_shape'] == [6144,1024], 'Wrong selected projection shape')
        for key in layer['kernel_hashes']:
            require(key in evidence['compiled_artifacts'], 'Layer lacks compiled artifact')
    for entry in evidence['compiled_artifacts'].values():
        for kind, record in entry['files'].items():
            require(sha(run/'candidate'/record['path']) == record['sha256'], 'Compiled artifact changed')
    numerical = []
    with np.load(run/'original/all-logits.npz', allow_pickle=False) as original, \
         np.load(run/'candidate/all-logits.npz', allow_pickle=False) as candidate:
        required = {f'step{i:02d}' for i in range(32)}
        require(set(original.files) == set(candidate.files) == required, 'Incomplete logit step keys')
        for step in range(32):
            left, right = original[f'step{step:02d}'], candidate[f'step{step:02d}']
            require(left.shape == right.shape == (1,151936) and left.dtype == right.dtype == np.float16, 'Wrong logit shape/dtype')
            a, b = left.astype(np.float32), right.astype(np.float32)
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            absolute = np.abs(b-a)
            allowed = 0.002+0.002*np.abs(a)
            violations = int((~(np.isfinite(a)&np.isfinite(b)&(absolute <= allowed))).sum())
            record = {'step': step, 'finite': finite, 'elements': left.size, 'violations': violations,
                      'max_abs_error': float(absolute.max()) if finite else None,
                      'max_allowed_error_ratio': float((absolute/allowed).max()) if finite else None}
            require(record == result['comparison']['steps'][step], 'Recomputed logit comparison differs at step '+str(step))
            numerical.append(record)
    require(workers['original']['raw_generated_token_ids'] == cfg['raw_expected_token_ids'], 'Fresh original tokens differ')
    exact_tokens = workers['candidate']['raw_generated_token_ids'] == cfg['raw_expected_token_ids']
    violations = sum(row['violations'] for row in numerical)
    elements = sum(row['elements'] for row in numerical)
    finite = all(row['finite'] for row in numerical)
    accepted = finite and violations == 0 and exact_tokens
    require(result['comparison']['pass'] == accepted and result['comparison']['violations'] == violations and
            result['comparison']['elements_checked'] == elements and result['comparison']['exact_32_tokens_equal'] == exact_tokens,
            'Controller comparison aggregate mismatch')
    require(result['status'] == ('GATE_UP_MODEL_CORRECTNESS_PASS' if accepted else 'GATE_UP_MODEL_CORRECTNESS_REJECTED'),
            'Controller accepted rejected numerical evidence')
    return {'status': 'MODEL_EVIDENCE_AUDIT_PASS', 'run_status': result['status'], 'model_candidate_accepted': accepted,
            'frozen_files_verified': len(cfg['frozen_files']), 'original_source_files_verified': len(cfg['source_manifest']),
            'fresh_processes': 2, 'all_step_logit_arrays_verified': 64, 'elements_compared': elements,
            'violations': violations, 'violation_fraction': violations/elements, 'all_finite': finite,
            'exact_32_tokens_equal': exact_tokens, 'prefill_logits_bitwise_equal': numerical[0]['max_abs_error'] == 0,
            'decode_steps_with_violations': [row['step'] for row in numerical if row['violations']],
            'maximum_absolute_logit_error': max(row['max_abs_error'] for row in numerical if row['finite']),
            'actual_decode_candidate_calls': 868, 'prepared_layers': 28,
            'compiled_specializations_verified': len(evidence['compiled_artifacts']),
            'runner_sha256': cfg['frozen_files']['runner.py'], 'candidate_source_sha256': cfg['candidate_source_sha256'],
            'result_sha256': sha(run/'result.json'), 'audit_source_sha256': sha(__file__),
            'audited_utc': datetime.now(timezone.utc).isoformat(),
            'limit': 'CPU audit of saved evidence; audit PASS validates the recorded rejection/acceptance, not candidate performance or broad model quality. No GPU run.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run/'gate-up-model-evidence-audit.json'
    require(not output.exists(), 'Preserve existing audit; choose another --output')
    result = audit(args.run)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
