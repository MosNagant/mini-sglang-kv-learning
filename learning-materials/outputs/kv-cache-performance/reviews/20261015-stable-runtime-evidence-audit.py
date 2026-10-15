"""Independent CPU-only recomputation of frozen 54 stable-runtime evidence."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok: raise AssertionError(message)
def near(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)


def bindings(run):
    result,cfg=read(run/'result.json'),read(run/'plan.json')
    require(sha(run/'plan.json')==result['plan_sha256'],'Plan hash changed')
    require(sha(run/'order.json')==result['order_sha256'],'Order hash changed')
    verified=[]
    for name,digest in cfg['frozen_files'].items():
        require(sha(run/name)==digest,'Frozen file changed: '+name);verified.append(name)
    for name,digest in cfg['source_manifest'].items():
        require(sha(run/'source'/name)==digest,'Original source changed: '+name);verified.append('source/'+name)
    tree=ast.parse((run/'stable_runtime.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTRACT' for t in n.targets))
    embedded=json.loads(ast.literal_eval(node.value.args[0]))
    require(embedded==cfg['contract'],'Embedded contract differs from plan')
    require(cfg['contract']['source_manifest']==cfg['source_manifest'],'Source manifest not bound to runtime')
    require(cfg['numerical_contract']=={'atol':.002,'rtol':.002,'all32logits':True,'exact32tokens':True},'Numerical gate changed')
    return result,cfg,verified


def check(run):
    result,cfg,verified=bindings(run)
    require(result['status']=='STABLE_RUNTIME_CORRECTNESS_PASS','Correctness run did not pass')
    expected=cfg['contract']['raw_expected_token_ids']
    require(len(expected)==32 and len(cfg['contract']['input_token_ids'])==128,'Fixed workload changed')
    order=read(run/'order.json');require(order==[{'policy':'eager','round':0},{'policy':'decode_graph','round':0}],'Strict order differs')
    records={}
    for policy in ('eager','decode_graph'):
        record=read(run/policy/'result.json');records[policy]=record
        require(record in result['workers'],'Worker not retained exactly in controller')
        require(record['status']=='STABLE_RUNTIME_LOGITS_COLLECTED','Worker incomplete')
        require(record['raw_generated_token_ids']==expected,'Token sequence differs')
        require(record['logits_count']==32 and record['graph_replays_observed']==(31 if policy=='decode_graph' else 0),'Actual logits/replays count differs')
        require(sha(run/policy/'all-logits.npz')==record['logits_sha256'],'Saved logits hash changed')
        require(record['logits_source_dtypes']==(['torch.float16']*32 if policy=='eager' else ['torch.float16']+['torch.float32']*31),'Upstream logit source dtype sequence differs')
    steps=[]
    with np.load(run/'eager/all-logits.npz',allow_pickle=False) as left,np.load(run/'decode_graph/all-logits.npz',allow_pickle=False) as right:
        require(set(left.files)==set(right.files)=={f'step{i:02d}' for i in range(32)},'Logit step set differs')
        for step in range(32):
            a,b=left[f'step{step:02d}'],right[f'step{step:02d}']
            require(a.shape==b.shape==(1,151936) and a.dtype==b.dtype==np.float32,'Saved logit metadata differs')
            finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
            error=np.abs(b-a)
            bad=~(np.isfinite(a)&np.isfinite(b)&(error<=.002+.002*np.abs(a)))
            precise_bad=~(np.isfinite(a)&np.isfinite(b)&(np.abs(b.astype(np.float64)-a)<=.002+.002*np.abs(a.astype(np.float64))))
            require(np.array_equal(bad,precise_bad),'Float32-vs-Float64 tolerance decision differs')
            steps.append({'step':step,'elements':a.size,'finite':finite,'violations':int(bad.sum()),
                          'max_abs_error':float(error.max()) if finite else None,
                          'equal_fp32_bit_patterns':bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)))})
    comparison=result['comparison']
    require(steps==comparison['steps'],'Saved per-step comparison differs')
    require(all(s['finite'] and s['violations']==0 for s in steps),'Strict logits rejected')
    require(comparison['elements_compared']==sum(s['elements'] for s in steps)==4861952,'Element count differs')
    require(comparison['violations']==0 and comparison['exact_32_tokens_equal'] and comparison['all_logits_within_tolerance'] and comparison['pass'],'Saved strict verdict differs')
    return {'mode':'check','frozen_files_verified':len(verified),'logit_arrays_verified':64,
            'logit_elements_compared':4861952,'all32_logits_bitwise_equal_after_value_preserving_cpu_conversion':all(s['equal_fp32_bit_patterns'] for s in steps),
            'max_abs_logit_difference':max(s['max_abs_error'] for s in steps),'actual_graph_replays':31,
            'exact_generated_tokens_per_policy':32,'steps':steps}


def timing(run):
    result,cfg,verified=bindings(run)
    require(result['status']=='STABLE_RUNTIME_TIMING_COLLECTION_COMPLETE','Timing collection incomplete')
    require(cfg['warmups_per_worker']==2 and cfg['samples_per_worker']==5 and cfg['fresh_rounds']==3,'Timing policy changed')
    require(result['fresh_workers']==6 and result['sample_count']==30,'Timing counts differ')
    accepted_name=Path(cfg['accepted_check']).name
    accepted=run.parents[1]/'m08-stable-runtime-check-results'/accepted_name
    require(read(accepted/'result.json')==read(run/'accepted-check.json'),'Accepted strict result changed')
    strict=check(accepted)
    old=read(accepted/'plan.json')
    for name in ('stable_runtime.py','runner.py','54-verify-stable-runtime.sh'):
        require(cfg['frozen_files'][name]==old['frozen_files'][name],'Code changed between correctness and timing')
    require(old['contract']==cfg['contract'],'Contract changed after correctness')
    expected=cfg['contract']['raw_expected_token_ids'];records={};samples=0
    order=read(run/'order.json')
    desired=[{'policy':p,'round':r} for r in (1,2,3) for p in (('decode_graph','eager') if r%2 else ('eager','decode_graph'))]
    require(order==desired,'Paired process order differs')
    for item in order:
        label=f"{item['policy']}-round{item['round']}";record=read(run/label/'result.json')
        require(record in result['workers'],'Worker not retained exactly')
        require(record['status']=='STABLE_RUNTIME_TIMING_WORKER_COMPLETE','Timing worker incomplete')
        require(record['policy']==item['policy'] and record['round']==item['round'],'Worker routing differs')
        require(len(record['samples'])==5,'Missing raw timing sample')
        values=[]
        for index,row in enumerate(record['samples']):
            require(row['sample']==index and row['raw_generated_steps']==32 and row['raw_generated_token_ids']==expected,'Timed tokens/index differs')
            duration=row['seconds'];require(math.isfinite(duration) and duration>0,'Invalid duration')
            require(near(row['raw_output_tokens_per_second'],32/duration),'Throughput derivation differs')
            values.append(duration);samples+=1
        require(near(statistics.median(values),record['median_seconds']),'Worker median differs')
        records[item['policy'],item['round']]=record
    pairs=[]
    for rnd in (1,2,3):
        e,g=[records[p,rnd]['median_seconds'] for p in ('eager','decode_graph')]
        pair={'round':rnd,'eager_ms':e*1000,'graph_ms':g*1000,'eager_over_graph':e/g,'graph_latency_change_pct':(g/e-1)*100}
        reported=result['pairs'][rnd-1];require(all(near(v,reported[k]) for k,v in pair.items()),'Paired summary differs');pairs.append(pair)
    require(result['all_rounds_at_least3pct_faster']==all(p['graph_latency_change_pct']<=-3 for p in pairs),'Signal threshold differs')
    return {'mode':'timing','frozen_files_verified':len(verified),'raw_samples_verified':samples,'fresh_workers_verified':6,
            'strict_check_recomputed':strict,'pairs':pairs,'all_rounds_at_least3pct_faster':result['all_rounds_at_least3pct_faster']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();cfg=read(run/'plan.json')
    evidence=check(run) if cfg['mode']=='check' else timing(run)
    report={'status':'STABLE_RUNTIME_INDEPENDENT_AUDIT_PASS','gpu_execution':False,'evidence':evidence,
            'audit_source_sha256':sha(__file__),'result_sha256':sha(run/'result.json'),'plan_sha256':sha(run/'plan.json'),
            'limit':'CPU recomputation of saved fixed128/32 evidence. Fair comparison is original Eager versus upstream original Graph through identical API; not a new Triton/kernel/AMD speedup or broad model-quality proof.'}
    path=run/'independent-evidence-audit.json';require(not path.exists(),'Preserve existing audit')
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
