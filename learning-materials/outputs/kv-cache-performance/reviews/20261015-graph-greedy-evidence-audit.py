"""Independent CPU-only hash, logits, routing and timing audit of experiment 55."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ARMS = ('original_graph', 'graph_greedy')
ANCHOR = "        require(kernel_api.store_cache is store_cache, 'Original KV entry has been replaced')\n"
HOOK = '        _EXPERIMENT_INSTALL_HOOK()  # 55: after source guards, before Engine\n'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise AssertionError(message)
def near(a, b): return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def bindings(run):
    cfg, result = read(run/'plan.json'), read(run/'result.json')
    require(sha(run/'plan.json') == result['plan_sha256'], 'Plan hash changed')
    require(sha(run/'order.json') == result['order_sha256'], 'Order hash changed')
    checked = []
    for name, digest in cfg['frozen_files'].items():
        require(sha(run/name) == digest, 'Frozen file changed: '+name); checked.append(name)
    for name, digest in cfg['source_manifest'].items():
        require(sha(run/'source'/name) == digest, 'Original source changed: '+name); checked.append('source/'+name)
    original = (run/'stable_runtime_original.py').read_text(encoding='utf-8')
    require(original.count(ANCHOR) == 1, '53 hook anchor differs')
    require((run/'experiment_runtime.py').read_text(encoding='utf-8') == original.replace(ANCHOR, ANCHOR+HOOK, 1), 'Unexpected runtime edit')
    parsed = ast.parse(original)
    node = next(n for n in parsed.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONTRACT' for t in n.targets))
    require(json.loads(ast.literal_eval(node.value.args[0])) == cfg['contract'], 'Embedded runtime contract differs')
    require(cfg['contract']['source_manifest'] == cfg['source_manifest'], 'Source manifest not bound to runtime')
    a = cfg['contract']['input_token_ids']; b = list(reversed(a))
    require(len(a) == 128 and a != b, 'Fixed input workload differs')
    require(cfg['strict_requests'] == [{'name':'A','input_token_ids':a},{'name':'B','input_token_ids':b},{'name':'A','input_token_ids':a}], 'Strict A/B/A sequence differs')
    require(cfg['warmups_per_worker'] == 2 and cfg['samples_per_worker'] == 5 and cfg['fresh_pairs'] == 3, 'Timing protocol differs')
    require(cfg['candidate_default'] is False and result['candidate_default'] is False, 'Unexpected automatic promotion')
    require(cfg['numerical_contract'] == 'all96 sampled logits bitwise FP32 and finite; exact32tokens/request; A repeats bitwise after B', 'Numeric contract changed')
    require(cfg['signal_rule'] == 'candidate median latency at least3% lower in each of3 fresh paired processes; engineering signal only, no automatic default promotion', 'Decision threshold changed')
    return cfg, result, checked


def routing(report, requests, arm):
    if arm == 'original_graph':
        require(report is None, 'Original unexpectedly has candidate routing'); return
    counts = report['counts']
    expected = {'capture_warmup_argmax':1,'captured_argmax':1,'replays':31*requests,
                'sample_hits':31*requests,'original_samples':requests,'unconsumed_replay_tickets':0}
    require(counts == expected, 'Candidate routing counts differ')
    require(report['fallback_reasons'] == {'no_replay_ticket':requests}, 'Unexpected fallback')
    require(not report['pending_ticket'] and report['installed'] and not report['candidate_default'], 'Patch lifecycle differs')


def strict(run):
    cfg, result, checked = bindings(run)
    require(cfg['mode'] == 'check' and result['status'] == 'GRAPH_GREEDY_CORRECTNESS_PASS', 'Strict result not accepted')
    require(read(run/'order.json') == [{'arm':a,'round':0} for a in ARMS], 'Strict order differs')
    expected = cfg['contract']['raw_expected_token_ids']
    require(len(expected) == 32, 'Output token contract differs')
    records, arrays = {}, {}
    for arm in ARMS:
        record = read(run/arm/'result.json'); records[arm] = record
        require(record in result['workers'], 'Worker not retained exactly')
        require(record['status'] == 'GRAPH_GREEDY_LOGITS_COLLECTED' and record['request_count'] == 3, 'Worker incomplete')
        require(record['graph_replays_observed'] == 93, 'Actual replay count differs')
        routing(record['candidate_routing'], 3, arm)
        for index, request in enumerate(record['requests']):
            entry = cfg['strict_requests'][index]
            require(request['request_index'] == index and request['name'] == entry['name'] and request['input_token_ids'] == entry['input_token_ids'], 'Request routing differs')
            require(len(request['raw_generated_token_ids']) == 32 and request['graph_replays_cumulative'] == 31*(index+1), 'Request token/replay count differs')
            require(request['logits_source_dtypes'] == ['torch.float16']+['torch.float32']*31, 'Source logit dtype sequence differs')
            if entry['name'] == 'A': require(request['raw_generated_token_ids'] == expected, 'A output changed')
            routing(request['candidate_routing'], index+1, arm)
            path = run/arm/request['logits_file']
            require(sha(path) == request['logits_sha256'], 'Saved logits hash changed')
            with np.load(path, allow_pickle=False) as values:
                require(set(values.files) == {f'step{i:02d}' for i in range(32)}, 'Missing saved logit steps')
                arrays[arm,index] = [values[f'step{i:02d}'] for i in range(32)]
    steps = []
    for index in range(3):
        require(records[ARMS[0]]['requests'][index]['raw_generated_token_ids'] == records[ARMS[1]]['requests'][index]['raw_generated_token_ids'], 'Output tokens disagree')
        for step, (a,b) in enumerate(zip(arrays[ARMS[0],index], arrays[ARMS[1],index])):
            require(a.shape == b.shape == (1,151936) and a.dtype == b.dtype == np.float32, 'Array metadata differs')
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            differences = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
            steps.append({'request_index':index,'name':cfg['strict_requests'][index]['name'],'step':step,
                          'finite':finite,'bit_differences':differences,
                          'max_abs_error':float(np.max(np.abs(a-b))) if finite else None})
    comparison = result['comparison']
    require(steps == comparison['steps'], 'Per-step saved comparison differs')
    require(all(row['finite'] and row['bit_differences'] == 0 for row in steps), 'Strict bitwise gate rejected')
    repeat = []
    changed_inputs = []
    for arm in ARMS:
        same = all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a,b in zip(arrays[arm,0],arrays[arm,2]))
        tokens_same = records[arm]['requests'][0]['raw_generated_token_ids'] == records[arm]['requests'][2]['raw_generated_token_ids']
        repeat.append({'arm':arm,'A_before_after_B_bitwise_equal':same,'A_before_after_B_tokens_equal':tokens_same})
        changed_inputs.append({'arm':arm,'A_vs_B_bit_differences':sum(int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))) for a,b in zip(arrays[arm,0],arrays[arm,1])),
                               'A_vs_B_tokens_differ':records[arm]['requests'][0]['raw_generated_token_ids'] != records[arm]['requests'][1]['raw_generated_token_ids']})
    require(repeat == comparison['A_B_A_repeat_checks'] and all(r['A_before_after_B_bitwise_equal'] and r['A_before_after_B_tokens_equal'] for r in repeat), 'A/B/A repeat changed')
    require(comparison['elements_compared'] == 3*32*151936 == 14585856 and comparison['bit_differences'] == 0 and comparison['exact_tokens_all_requests'] and comparison['pass'], 'Strict summary differs')
    return {'mode':'check','frozen_files_verified':len(checked),'saved_logit_arrays_verified':192,
            'logit_elements_compared_between_arms':14585856,'bit_differences':0,'generated_tokens_per_arm':96,
            'replays_per_arm':93,'candidate_sample_hits':93,'A_B_A_repeat_checks':repeat,
            'changed_input_observations':changed_inputs}


def timing(run):
    cfg, result, checked = bindings(run)
    require(cfg['mode'] == 'timing' and result['status'] == 'GRAPH_GREEDY_TIMING_COLLECTION_COMPLETE', 'Timing incomplete')
    require(result['fresh_workers'] == 6 and result['samples'] == 30, 'Timing count differs')
    accepted = run.parents[1]/'m08-graph-greedy-check-results'/Path(cfg['accepted_check']).name
    require(read(accepted/'result.json') == read(run/'accepted-check.json'), 'Accepted strict result changed')
    strict_result = strict(accepted)
    prior = read(accepted/'plan.json')
    for name,digest in prior['frozen_files'].items(): require(cfg['frozen_files'][name] == digest, 'Code changed after strict check: '+name)
    for key in ('contract','strict_requests','numerical_contract','signal_rule'): require(cfg[key] == prior[key], 'Protocol changed after strict check')
    desired = [{'arm':arm,'round':rnd} for rnd in (1,2,3) for arm in (ARMS[::-1] if rnd%2 else ARMS)]
    require(read(run/'order.json') == desired, 'Paired order differs')
    expected = cfg['contract']['raw_expected_token_ids']; records = {}; samples = 0
    for item in desired:
        arm, rnd = item['arm'], item['round']
        record = read(run/f'{arm}-round{rnd}'/'result.json')
        require(record in result['workers'], 'Worker not retained exactly')
        require(record['status'] == 'GRAPH_GREEDY_TIMING_WORKER_COMPLETE' and record['arm'] == arm and record['round'] == rnd, 'Worker identity/completion differs')
        require(len(record['samples']) == 5 and record['graph_replays_observed'] == 217, 'Worker counts differ')
        routing(record['candidate_routing'], 7, arm)
        values = []
        for index,row in enumerate(record['samples']):
            require(row['sample'] == index and row['raw_generated_token_ids'] == expected, 'Sample output differs')
            require(row['graph_replays_cumulative'] == 31*(index+3), 'Sample replay counts differ')
            routing(row['candidate_routing'], index+3, arm)
            seconds = row['seconds']; require(math.isfinite(seconds) and seconds>0, 'Invalid duration')
            require(near(row['raw_output_tokens_per_second'],32/seconds), 'Throughput differs')
            values.append(seconds); samples+=1
        require(near(statistics.median(values),record['median_seconds']), 'Median differs')
        records[arm,rnd] = record
    pairs = []
    for rnd in (1,2,3):
        a,b = [records[arm,rnd]['median_seconds'] for arm in ARMS]
        row = {'round':rnd,'original_graph_ms':a*1000,'graph_greedy_ms':b*1000,
               'original_over_candidate':a/b,'latency_reduction_pct':(1-b/a)*100}
        require(all(near(value,result['pairs'][rnd-1][key]) for key,value in row.items()), 'Paired timing derivation differs')
        pairs.append(row)
    signal = all(row['latency_reduction_pct']>=3 for row in pairs)
    require(result['all_rounds_at_least3pct_faster'] == signal, 'Signal decision differs')
    return {'mode':'timing','frozen_files_verified':len(checked),'fresh_workers_verified':6,'raw_samples_verified':samples,
            'strict_check_recomputed':strict_result,'pairs':pairs,'predeclared_signal_met':signal}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    evidence=strict(run) if read(run/'plan.json')['mode']=='check' else timing(run)
    report={'status':'GRAPH_GREEDY_INDEPENDENT_AUDIT_PASS','gpu_execution':False,'evidence':evidence,
            'audit_source_sha256':sha(__file__),'result_sha256':sha(run/'result.json'),'plan_sha256':sha(run/'plan.json'),
            'limit':'CPU audit of saved fixed B1/TP1/FP16 Qwen3-0.6B/RTX4070Ti evidence. Candidate retains original math and captures original sampling; not kernel fusion, Triton/AMD gain or general model-quality proof. No automatic promotion.'}
    target=run/'independent-evidence-audit.json';require(not target.exists(),'Preserve previous audit')
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
