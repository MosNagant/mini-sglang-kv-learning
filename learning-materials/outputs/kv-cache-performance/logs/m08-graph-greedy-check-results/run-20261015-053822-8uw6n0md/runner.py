"""Bounded original-Graph vs original-Graph-plus-greedy-capture experiment.

Strict phase retains all32 logits/tokens for A -> B -> A in one runtime per arm.
Only bitwise-FP32 and exact-token PASS enables three fresh timing pairs, five
fixed-A samples each after two warmups. Every round must improve median latency
by >=3% to meet the predeclared engineering signal. Never change53's default.
The new candidate moves original argmax/cast submissions inside the same Graph;
it does not fuse device kernels or change model arithmetic.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/weltlaw/projects/mini-sglang-kv')
ARMS = ('original_graph', 'graph_greedy')
INSERTION_ANCHOR = "        require(kernel_api.store_cache is store_cache, 'Original KV entry has been replaced')\n"


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(condition, message):
    if not condition: raise RuntimeError(message)


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def runtime_with_hook(source):
    """One explicit post-import/pre-Engine hook; all other53 text unchanged."""
    require(source.count(INSERTION_ANCHOR)==1,'Runtime hook anchor differs')
    return source.replace(INSERTION_ANCHOR,INSERTION_ANCHOR+'        _EXPERIMENT_INSTALL_HOOK()  # 55: after source guards, before Engine\n',1)


def validate_frozen(run):
    cfg=read(run/'plan.json')
    for name,digest in cfg['frozen_files'].items(): require(sha(run/name)==digest,'Frozen file changed: '+name)
    for name,digest in cfg['source_manifest'].items(): require(sha(run/'source'/name)==digest,'Source changed: '+name)
    original=(run/'stable_runtime_original.py').read_text(encoding='utf-8')
    require((run/'experiment_runtime.py').read_text(encoding='utf-8')==runtime_with_hook(original),'Unexpected runtime edit')
    runtime=load(run/'experiment_runtime.py','frozen_graph_greedy_runtime')
    require(runtime.CONTRACT==cfg['contract'],'Runtime contract changed')
    greedy=load(run/'graph_greedy.py','frozen_graph_greedy_patch')
    for name,digest in greedy.SOURCE_SHA256.items():
        require(cfg['source_manifest']['python/minisgl/engine/'+name]==digest,'Patch source pin differs: '+name)
    return cfg,runtime,greedy


def routing(handle, requests):
    if handle is None: return None
    report=handle.assert_usage(replays=31*requests,sample_hits=31*requests,original_samples=requests)
    require(report['fallback_reasons']=={'no_replay_ticket':requests},'Unexpected candidate fallback reason')
    return report


def worker(run, mode, arm, round_id):
    import numpy as np
    out=run/(arm if mode=='check' else f'{arm}-round{round_id}')
    out.mkdir()
    result={'status':'FAILED','mode':mode,'arm':arm,'round':round_id,'requests':[],'samples':[]}
    runtime=None
    owned={'handle':None}
    replay_owner=None
    original_replay=None
    try:
        cfg,runtime_module,greedy=validate_frozen(run)
        def hook():
            if arm=='graph_greedy': owned['handle']=greedy.install()
        runtime_module._EXPERIMENT_INSTALL_HOOK=hook
        runtime=runtime_module.StableRuntime(execution='decode_graph',project=run/'source',model_path=cfg['model_path'])
        result.update(environment=runtime.environment,source_validation=runtime.validation)
        handle=owned['handle']
        require((handle is not None)==(arm=='graph_greedy'),'Candidate install path differs')
        replay_owner=runtime.model.engine.graph_runner
        original_replay=replay_owner.replay
        replay_count=0
        def counted_replay(batch):
            nonlocal replay_count
            replay_count+=1
            return original_replay(batch)
        # Identical lightweight observation in BOTH arms, including timing.
        replay_owner.replay=counted_replay
        expected=cfg['contract']['raw_expected_token_ids']
        if mode=='check':
            for request_index,entry in enumerate(cfg['strict_requests']):
                logits,dtypes=[],[]
                sampler=runtime.model.engine.sampler
                original_sample=sampler.sample
                def observed_sample(value,args):
                    require(tuple(value.shape)==(1,151936),'Sampled logits shape differs')
                    logits.append(value.detach().clone())
                    dtypes.append(str(value.dtype))
                    return original_sample(value,args)  # original view identity/value unchanged
                sampler.sample=observed_sample
                try:
                    output=runtime.generate(entry['input_token_ids'])
                    runtime.synchronize()
                finally:
                    sampler.sample=original_sample
                require(len(logits)==len(runtime.raw_token_ids)==32,'Incomplete request')
                require(dtypes==['torch.float16']+['torch.float32']*31,'Expected original prefill FP16 + Graph decode FP32')
                require(replay_count==31*(request_index+1),'Actual Graph replay count differs')
                evidence=routing(handle,request_index+1)
                arrays={f'step{i:02d}':value.cpu().numpy().astype(np.float32) for i,value in enumerate(logits)}
                filename=f'request{request_index}-{entry["name"]}-logits.npz'
                np.savez(out/filename,**arrays)
                raw=list(runtime.raw_token_ids)
                if entry['name']=='A': require(raw==expected,'A differs from historical accepted tokens')
                result['requests'].append({'request_index':request_index,'name':entry['name'],
                    'input_token_ids':entry['input_token_ids'],'raw_generated_token_ids':raw,'output':output,
                    'logits_file':filename,'logits_sha256':sha(out/filename),'logits_source_dtypes':dtypes,
                    'graph_replays_cumulative':replay_count,'candidate_routing':evidence})
                save(out/'result.json',result)
            result.update(status='GRAPH_GREEDY_LOGITS_COLLECTED',request_count=3,
                candidate_routing=routing(handle,3),graph_replays_observed=replay_count,
                limit='A/B/A observed requests only; CPU snapshots FP32. No timed observer or broad model quality claim.')
        else:
            for index in range(2):
                runtime.generate()
                runtime.synchronize()
                require(runtime.raw_token_ids==expected,'Warmup tokens differ')
                routing(handle,index+1)
            runtime._torch.cuda.reset_peak_memory_stats()
            for index in range(5):
                runtime.synchronize()
                started=time.perf_counter()
                output=runtime.generate()
                runtime.synchronize()
                seconds=time.perf_counter()-started
                require(replay_count==31*(index+3),'Timed Graph replay count differs')
                evidence=routing(handle,index+3)
                record={'sample':index,'seconds':seconds,'raw_generated_token_ids':list(runtime.raw_token_ids),
                    'output':output,'graph_replays_cumulative':replay_count,'candidate_routing':evidence,
                    'raw_output_tokens_per_second':32/seconds if seconds>0 else None}
                result['samples'].append(record)
                save(out/'result.json',result)
                require(math.isfinite(seconds) and seconds>0 and runtime.raw_token_ids==expected,'Timed duration/tokens invalid')
            result.update(status='GRAPH_GREEDY_TIMING_WORKER_COMPLETE',
                median_seconds=statistics.median(row['seconds'] for row in result['samples']),
                peak_allocated_bytes=runtime._torch.cuda.max_memory_allocated(),
                candidate_routing=routing(handle,7),graph_replays_observed=replay_count,
                measurement='perf_counter around persistent53 API generate+GPU completion; guards/scheduler/sampling/raw-token recording/detokenize included; source/model verification/load/JIT/capture/warmups excluded')
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        if replay_owner is not None and original_replay is not None: replay_owner.replay=original_replay
        if runtime is not None:
            try: runtime.close()
            except BaseException:
                result['shutdown_error']=traceback.format_exc();result['status']='FAILED'
        if owned['handle'] is not None:
            result['candidate_final_report']=owned['handle'].report()
            try: owned['handle'].uninstall()
            except BaseException:
                result['uninstall_error']=traceback.format_exc();result['status']='FAILED'
        save(out/'result.json',result)
    require(result['status'] in ('GRAPH_GREEDY_LOGITS_COLLECTED','GRAPH_GREEDY_TIMING_WORKER_COMPLETE'),'Worker cleanup failed')


def compare_logits(run, cfg):
    import numpy as np
    records={arm:read(run/arm/'result.json') for arm in ARMS}
    require(all(row['status']=='GRAPH_GREEDY_LOGITS_COLLECTED' and len(row['requests'])==3 for row in records.values()),'Strict workers incomplete')
    steps=[]
    tokens_equal=True
    total=0
    for index,entry in enumerate(cfg['strict_requests']):
        left,right=[records[arm]['requests'][index] for arm in ARMS]
        for arm,row in zip(ARMS,(left,right)):
            require(row['name']==entry['name'] and row['input_token_ids']==entry['input_token_ids'],'Strict request differs')
            require(sha(run/arm/row['logits_file'])==row['logits_sha256'],'Saved logits changed')
        same_tokens=left['raw_generated_token_ids']==right['raw_generated_token_ids']
        tokens_equal &= same_tokens
        if entry['name']=='A': tokens_equal &= left['raw_generated_token_ids']==cfg['contract']['raw_expected_token_ids']
        with np.load(run/ARMS[0]/left['logits_file'],allow_pickle=False) as a, np.load(run/ARMS[1]/right['logits_file'],allow_pickle=False) as b:
            require(set(a.files)==set(b.files)=={f'step{i:02d}' for i in range(32)},'Incomplete saved step set')
            for step in range(32):
                x,y=a[f'step{step:02d}'],b[f'step{step:02d}']
                require(x.shape==y.shape==(1,151936) and x.dtype==y.dtype==np.float32,'Saved logits metadata differs')
                finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
                differences=int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32)))
                total+=x.size
                steps.append({'request_index':index,'name':entry['name'],'step':step,'finite':finite,
                    'bit_differences':differences,'max_abs_error':float(np.max(np.abs(x-y))) if finite else None})
    repeat=[]
    for arm in ARMS:
        reqs=records[arm]['requests']
        with np.load(run/arm/reqs[0]['logits_file']) as a,np.load(run/arm/reqs[2]['logits_file']) as b:
            same=all(np.array_equal(a[key].view(np.uint32),b[key].view(np.uint32)) for key in a.files)
        repeat.append({'arm':arm,'A_before_after_B_bitwise_equal':same,
                       'A_before_after_B_tokens_equal':reqs[0]['raw_generated_token_ids']==reqs[2]['raw_generated_token_ids']})
    passes=bool(tokens_equal and all(s['finite'] and s['bit_differences']==0 for s in steps)
                and all(r['A_before_after_B_bitwise_equal'] and r['A_before_after_B_tokens_equal'] for r in repeat))
    return {'pass':passes,'contract':'every element bitwise FP32 equality including signed zero, all finite; exact32tokens/request',
        'requests':3,'steps':steps,'elements_compared':total,'bit_differences':sum(row['bit_differences'] for row in steps),
        'exact_tokens_all_requests':bool(tokens_equal),'A_B_A_repeat_checks':repeat,
        'B_reference':'same-session original whole-decode Graph; fixed reversed128 input IDs; no presumed text-quality target'}


def prepare_run(args,run):
    runtime_file=HERE/'53-m08-stable-runtime.py'
    runtime=load(runtime_file,'graph_greedy_original_runtime_source')
    verified=runtime.verify_environment(args.project,args.model_path)
    original=runtime_file.read_text(encoding='utf-8')
    modified=runtime_with_hook(original)
    shutil.copyfile(runtime_file,run/'stable_runtime_original.py')
    (run/'experiment_runtime.py').write_text(modified,encoding='utf-8')
    (run/'runtime-hook.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='original53',tofile='55-controlled-runtime')),encoding='utf-8')
    shutil.copyfile(HERE/'stable/graph_greedy.py',run/'graph_greedy.py')
    shutil.copyfile(__file__,run/'runner.py')
    shutil.copyfile(HERE/'55-verify-graph-greedy.sh',run/'55-verify-graph-greedy.sh')
    for name,digest in runtime.CONTRACT['source_manifest'].items():
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.project/name,target)
        require(sha(target)==digest,'Original source changed during copy: '+name)
    a=list(runtime.CONTRACT['input_token_ids']);b=list(reversed(a))
    require(a!=b,'B prompt must differ')
    runtime.validate_inputs(b)
    frozen=('stable_runtime_original.py','experiment_runtime.py','runtime-hook.diff','graph_greedy.py','runner.py','55-verify-graph-greedy.sh')
    cfg={'mode':args.mode,'contract':runtime.CONTRACT,'model_path':verified['model_path'],
        'source_manifest':runtime.CONTRACT['source_manifest'],'validated_environment':verified,
        'frozen_files':{name:sha(run/name) for name in frozen},
        'strict_requests':[{'name':'A','input_token_ids':a},{'name':'B','input_token_ids':b},{'name':'A','input_token_ids':a}],
        'numerical_contract':'all96 sampled logits bitwise FP32 and finite; exact32tokens/request; A repeats bitwise after B',
        'warmups_per_worker':2,'samples_per_worker':5,'fresh_pairs':3,
        'signal_rule':'candidate median latency at least3% lower in each of3 fresh paired processes; engineering signal only, no automatic default promotion',
        'method':'unchanged original whole-decode Graph and FP32 logits; capture original torch.argmax plus int32 conversion; CPU submission reduction, not fewer device kernels',
        'candidate_default':False}
    if args.mode=='timing':
        require(args.accepted_check is not None,'Timing requires --accepted-check')
        accepted=args.accepted_check.resolve(strict=True)
        prior,prior_cfg=read(accepted/'result.json'),read(accepted/'plan.json')
        require(prior['status']=='GRAPH_GREEDY_CORRECTNESS_PASS' and prior['comparison']['pass'],'Strict candidate was not accepted')
        require(prior['plan_sha256']==sha(accepted/'plan.json'),'Accepted plan changed')
        for key in ('contract','source_manifest','frozen_files','strict_requests','numerical_contract','signal_rule'):
            require(prior_cfg[key]==cfg[key],'Correctness/timing protocol differs: '+key)
        require(compare_logits(accepted,prior_cfg)==prior['comparison'],'Accepted arrays/comparison differ')
        for arm in ARMS: require(read(accepted/arm/'result.json') in prior['workers'],'Accepted worker differs')
        shutil.copyfile(accepted/'result.json',run/'accepted-check.json')
        cfg['frozen_files']['accepted-check.json']=sha(run/'accepted-check.json')
        cfg['accepted_check']=str(accepted)
    save(run/'plan.json',cfg)
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('check','timing'),required=True)
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--accepted-check',type=Path)
    parser.add_argument('--worker-run',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--round',type=int,default=0)
    args=parser.parse_args()
    if args.worker_run:
        require(args.arm is not None,'Worker arm required')
        worker(args.worker_run.resolve(),args.mode,args.arm,args.round)
        return
    args.project=args.project.resolve(strict=True)
    parent=args.project/'experiments'/('m08-graph-greedy-'+args.mode);parent.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='run-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'),dir=parent))
    print('RUN_DIRECTORY='+str(run),flush=True)
    result,primary={'status':'FAILED','mode':args.mode,'workers':[]},None
    try:
        cfg=prepare_run(args,run);result['plan_sha256']=sha(run/'plan.json')
        order=([{'arm':arm,'round':0} for arm in ARMS] if args.mode=='check' else
               [{'arm':arm,'round':rnd} for rnd in (1,2,3) for arm in (ARMS[::-1] if rnd%2 else ARMS)])
        save(run/'order.json',order);result['order_sha256']=sha(run/'order.json')
        for item in order:
            label=item['arm'] if args.mode=='check' else f"{item['arm']}-round{item['round']}"
            print('RUN '+label,flush=True)
            with (run/(label+'.log')).open('xb') as log:
                proc=subprocess.run([sys.executable,'-B','-u',str(run/'runner.py'),'--mode',args.mode,
                    '--worker-run',str(run),'--arm',item['arm'],'--round',str(item['round'])],
                    cwd=args.project,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            require(proc.returncode==0,'Worker failed; retained '+label+'.log')
            record=read(run/label/'result.json');result['workers'].append(record);save(run/'result.json',result)
            if args.mode=='timing': print(f"COLLECTED {label} median_ms={record['median_seconds']*1000:.6f}",flush=True)
        if args.mode=='check':
            comparison=compare_logits(run,cfg)
            result.update(status='GRAPH_GREEDY_CORRECTNESS_PASS' if comparison['pass'] else 'GRAPH_GREEDY_CORRECTNESS_REJECTED',
                          comparison=comparison,fresh_workers=2)
        else:
            pairs=[]
            for rnd in (1,2,3):
                rows={r['arm']:r for r in result['workers'] if r['round']==rnd}
                original,candidate=[rows[arm]['median_seconds'] for arm in ARMS]
                pairs.append({'round':rnd,'original_graph_ms':original*1000,'graph_greedy_ms':candidate*1000,
                              'original_over_candidate':original/candidate,'latency_reduction_pct':(1-candidate/original)*100})
            result.update(status='GRAPH_GREEDY_TIMING_COLLECTION_COMPLETE',fresh_workers=6,samples=30,pairs=pairs,
                          all_rounds_at_least3pct_faster=all(row['latency_reduction_pct']>=3 for row in pairs))
        result.update(method=cfg['method'],candidate_default=False,
            limit='Fixed128/32 B1 TP1 greedy; compares originalGraph against sameGraph plus captured sampling. No Triton/AMD/service/broad-quality speed claim;53 remains default unless separately promoted.')
    except BaseException:
        primary=traceback.format_exc();result['error']=primary
        (run/'controller-error.txt').write_text(primary,encoding='utf-8')
        raise
    finally:
        save(run/'result.json',result)
        shared=HERE/'logs'/('m08-graph-greedy-'+args.mode+'-results')/run.name
        try:
            shared.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(run,shared)
            print('SHARED_RUN_DIRECTORY='+str(shared),flush=True)
        except BaseException:
            print('SHARED_COPY_FAILED; local evidence retained: '+str(run),flush=True)
            if primary is None: raise
    print('STATUS='+result['status'],flush=True)


if __name__=='__main__': main()
