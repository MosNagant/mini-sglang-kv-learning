"""CPU corruption controls for the independent stable-runtime evidence audit."""
import copy
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name('20261015-stable-runtime-evidence-audit.py')
spec = importlib.util.spec_from_file_location('stable_independent_audit', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
STRICT = BASE/'logs/m08-stable-runtime-check-results/run-20261015-052859-ufwymynd'
TIMING = BASE/'logs/m08-stable-runtime-timing-results/run-20261015-053033-nlwp0fei'


def main():
    results = []
    actual_read = audit.read
    # Only in-memory copies are corrupted. Saved experimental files stay intact.
    def rejected(name, run, function, change):
        def altered(path):
            value = copy.deepcopy(actual_read(path))
            change(Path(path), value)
            return value
        audit.read = altered
        try:
            function(run)
        except AssertionError as exc:
            results.append({'name': name, 'rejected': True, 'reason': str(exc)})
        else:
            raise AssertionError('Corruption was accepted: ' + name)
        finally:
            audit.read = actual_read

    def wrong_plan_hash(path, data):
        if path == STRICT/'result.json': data['plan_sha256'] = '0'*64
    rejected('changed_plan_hash', STRICT, audit.check, wrong_plan_hash)

    def wrong_source_hash(path, data):
        if path == STRICT/'plan.json':
            name = next(iter(data['source_manifest']))
            data['source_manifest'][name] = '0'*64
    rejected('changed_frozen_original_source_hash', STRICT, audit.check, wrong_source_hash)

    def wrong_saved_logits_hash(path, data):
        if path == STRICT/'decode_graph/result.json': data['logits_sha256'] = '0'*64
        if path == STRICT/'result.json':
            for worker in data['workers']:
                if worker['policy'] == 'decode_graph': worker['logits_sha256'] = '0'*64
    rejected('changed_saved_logits_hash', STRICT, audit.check, wrong_saved_logits_hash)

    def wrong_summary(path, data):
        if path == STRICT/'result.json': data['comparison']['steps'][0]['max_abs_error'] = 1.0
    rejected('invented_logit_summary', STRICT, audit.check, wrong_summary)

    def wrong_warmups(path, data):
        if path == TIMING/'plan.json': data['warmups_per_worker'] = 1
    rejected('unequal_timing_warmup_policy', TIMING, audit.timing, wrong_warmups)

    def mutate_all_records(value, key, mutator):
        if isinstance(value, dict):
            if key in value: mutator(value)
            for child in list(value.values()): mutate_all_records(child, key, mutator)
        elif isinstance(value, list):
            for child in value: mutate_all_records(child, key, mutator)

    def bad_seconds(path, data):
        if path.name == 'result.json' and TIMING in path.parents:
            mutate_all_records(data, 'seconds', lambda item: item.update(seconds=float('nan')))
    rejected('nonfinite_raw_time_even_when_controller_matches', TIMING, audit.timing, bad_seconds)

    def bad_median(path, data):
        if path.name == 'result.json' and TIMING in path.parents:
            mutate_all_records(data, 'median_seconds', lambda item: item.update(median_seconds=item['median_seconds']+1))
    rejected('invented_worker_medians_even_when_controller_matches', TIMING, audit.timing, bad_median)

    def bad_tokens(path, data):
        if path.name == 'result.json' and TIMING in path.parents:
            def corrupt(item): item['raw_generated_token_ids'][0] += 1
            mutate_all_records(data, 'raw_generated_token_ids', corrupt)
    rejected('wrong_timed_tokens_even_when_controller_matches', TIMING, audit.timing, bad_tokens)

    result = {'status': 'STABLE_AUDIT_CPU_CONTROLS_PASS', 'gpu_execution': False,
              'controls': results, 'count': len(results),
              'audit_source_sha256': audit.sha(SOURCE),
              'controls_source_sha256': audit.sha(__file__),
              'limit': 'In-memory corruption controls for saved-evidence auditing; no GPU or model execution.'}
    target = Path(__file__).with_suffix('.json')
    if target.exists(): raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
