"""Frozen seven-arm Gate/Up experiment: real fixtures, numeric gates, paired timing.

Coach implementation. No baseline changes; all failed candidates/results retained.
The primary target is a 28-layer sequence of F.linear + FlashInfer SiLU*Up,
not GEMV alone. Every timing worker is a fresh process. Graph and Eager differ.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/weltlaw/projects/mini-sglang-kv')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, obj): Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
def append(path, obj):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')
        handle.flush()
def need(ok, message):
    if not ok: raise RuntimeError(message)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def numerical_check(actual, reference, oracle):
    import numpy as np
    a, r, o = (np.asarray(t, dtype=np.float64) for t in (actual, reference, oracle))
    if a.shape != r.shape or a.shape != o.shape or not all(np.isfinite(t).all() for t in (a,r,o)):
        return {'pass': False, 'reason': 'shape or finite check failed'}
    delta = np.abs(a-r)
    bad = delta > (.002 + .002*np.abs(r))
    scale = max(float(np.linalg.norm(o)), 1e-12)
    ae, re = float(np.linalg.norm(a-o))/scale, float(np.linalg.norm(r-o))/scale
    return {'pass': bool(not bad.any() and ae <= 1.10*re+1e-6),
        'violations': int(bad.sum()), 'max_abs_vs_original': float(delta.max()),
        'rmse_vs_original': float(np.sqrt(np.mean(delta*delta))),
        'candidate_nrmse': ae, 'original_nrmse': re, 'nrmse_limit': 1.10*re+1e-6,
        'worst_original_index': int(delta.argmax()), 'element_count': int(a.size)}

def setup(run):
    import numpy as np
    import torch
    cfg = read(run/'plan.json')
    for name, digest in cfg['frozen_files'].items(): need(sha(run/name)==digest, 'Frozen source changed: '+name)
    fixture_run = Path(cfg['fixture_run'])
    need(sha(fixture_run/'fixture.json')==cfg['fixture_sha256'], 'Fixture manifest changed')
    fixture = read(fixture_run/'fixture.json')
    need(fixture['status']=='GATE_UP_FIXTURE_CAPTURE_PASS', 'Fixture capture incomplete')
    env = {'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda}
    need(all(env[k]==fixture['environment'][k] for k in env), 'GPU environment differs')
    need({k:importlib.metadata.version(k) for k in fixture['environment']['packages']}==fixture['environment']['packages'], 'Packages differ')
    for item in fixture['weights']+fixture['samples']:
        need(sha(fixture_run/item['path'])==item['sha256'], 'Fixture file changed')
    mod=load(run/'candidates.py','frozen_gate_up_candidates')
    need(mod.candidates()==cfg['candidates'], 'Candidate declarations changed')
    weights = {w['layer']: np.load(fixture_run/w['path'],allow_pickle=False) for w in fixture['weights']}
    samples = {(s['step'],s['layer']): dict(np.load(fixture_run/s['path'],allow_pickle=False)) for s in fixture['samples']}
    return cfg, mod, weights, samples

def compiler_dump(plan, folder):
    folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for i,k in enumerate(plan.get('compiled_kernels',[])):
        if k is None: continue
        item={'kernel_index':i,'name':k.name,'n_regs':k.n_regs,'n_spills_raw':k.n_spills,'shared_bytes':k.metadata.shared,'files':{}}
        for suffix in ('ttir','ttgir','llir','ptx','cubin'):
            value=k.asm.get(suffix)
            if value is None: continue
            p=folder/f'kernel{i}.{suffix}'
            p.write_bytes(value if isinstance(value,bytes) else value.encode())
            item['files'][p.name]=sha(p)
        entries.append(item)
    save(folder/'compiled.json',entries)

def validate(run):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from flashinfer import silu_and_mul
    cfg,mod,weights,samples=setup(run)
    directory=run/'numerical';directory.mkdir()
    records=[]
    eligibility={x['name']:True for x in cfg['candidates']}
    for layer in range(28):
        wcpu=weights[layer]
        w=torch.from_numpy(wcpu.copy()).cuda()
        wbefore=w.clone()
        w64=wcpu.astype(np.float64)
        for step in (1,16,31):
            s=samples[step,layer]
            x=torch.from_numpy(s['x'].copy()).cuda();xbefore=x.clone()
            z=F.linear(x,w);y=silu_and_mul(z)
            torch.cuda.synchronize()
            zcpu,ycpu=z.cpu().numpy(),y.cpu().numpy()
            need(np.array_equal(zcpu,s['z']) and np.array_equal(ycpu,s['y']),f'Fresh baseline differs step{step} layer{layer}')
            z64=s['x'].astype(np.float64)@w64.T
            rounded=z64.astype(np.float16).astype(np.float64)
            gate,up=np.split(rounded,2,axis=1)
            # Stable mathematical SiLU; FP16 intermediate boundary retained.
            sigmoid=np.empty_like(gate);positive=gate>=0
            sigmoid[positive]=1/(1+np.exp(-gate[positive]))
            eg=np.exp(gate[~positive]);sigmoid[~positive]=eg/(1+eg)
            y64=gate*sigmoid*up
            np.savez(directory/f'oracle-step{step:02d}-layer{layer:02d}.npz',z64=z64,y64=y64)
            for spec in cfg['candidates']:
                name=spec['name'];record={'name':name,'step':step,'layer':layer,'pass':False}
                target=directory/name;target.mkdir(exist_ok=True)
                try:
                    plan=mod.prepare(name,x,w)
                    value=plan['call']()
                    need(value.dtype==torch.float16,'Candidate output dtype differs')
                    cy=silu_and_mul(value) if plan['kind']=='gemv' else value
                    torch.cuda.synchronize()
                    arrays={'y':cy.cpu().numpy()}
                    checks={'y':numerical_check(arrays['y'],s['y'],y64)}
                    if plan['kind']=='gemv':
                        arrays['z']=value.cpu().numpy();checks['z']=numerical_check(arrays['z'],s['z'],z64)
                    packed=plan['packed_weight']
                    roundtrip=packed is None or torch.equal(packed.permute(1,0,2).contiguous().view(6144,1024),w)
                    readonly=torch.equal(x,xbefore) and torch.equal(w,wbefore)
                    record.update(checks=checks,readonly=readonly,packing_roundtrip=bool(roundtrip),
                        setup_seconds=plan['setup_seconds'],workspace_bytes=plan['workspace_bytes'],prepack_bytes=plan['prepack_bytes'])
                    record['pass']=bool(readonly and roundtrip and all(c['pass'] for c in checks.values()))
                    np.savez(target/f'step{step:02d}-layer{layer:02d}.npz',**arrays)
                    if layer==0 and step==1: compiler_dump(plan,target/'compiler')
                except Exception:
                    record['error']=traceback.format_exc()
                records.append(record);eligibility[name] &= record['pass']
                append(directory/'cases.jsonl',record)
                save(directory/'progress.json',{'completed':len(records),'last':record,'eligible_so_far':eligibility})
            need(torch.equal(x,xbefore) and torch.equal(w,wbefore),'Fixture corrupted')
        print(f'NUMERIC_LAYER_DONE={layer+1}/28',flush=True)
    result={'status':'NUMERICAL_CAMPAIGN_COMPLETE','case_count':len(records),'records':records,
            'eligible':[name for name,ok in eligibility.items() if ok],
            'failed':[name for name,ok in eligibility.items() if not ok]}
    save(run/'numerical.json',result)
    print('NUMERIC_ELIGIBLE='+str(result['eligible']),flush=True)

def timing(run, task):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from flashinfer import silu_and_mul
    cfg,mod,weights,samples=setup(run)
    t=read(task);folder=task.parent
    names=t['arms'];step=t['step']
    w=[torch.from_numpy(weights[i].copy()).cuda() for i in range(28)]
    x=[torch.from_numpy(samples[step,i]['x'].copy()).cuda() for i in range(28)]
    before_w=[a.clone() for a in w];before_x=[a.clone() for a in x]
    plans={};functions={};graphs={};outputs={};plan_info={}
    for name in names:
        if name=='original':
            functions[name]=[(lambda i=i:silu_and_mul(F.linear(x[i],w[i]))) for i in range(28)]
            plan_info[name]={'kind':'original F.linear + FlashInfer activation','allocation_policy':'original allocating APIs'}
        else:
            plans[name]=[mod.prepare(name,x[i],w[i]) for i in range(28)]
            functions[name]=[(lambda p=p:silu_and_mul(p['call']())) if p['kind']=='gemv' else p['call'] for p in plans[name]]
            plan_info[name]={'setup_seconds_sum':sum(p['setup_seconds'] for p in plans[name]),
                'prepack_bytes_sum':sum(p['prepack_bytes'] for p in plans[name]),
                'workspace_bytes_sum':sum(p['workspace_bytes'] for p in plans[name]),
                'allocation_policy':'reused candidate output/workspace; gemv arms use original allocating activation'}
        for f in functions[name]:
            for _ in range(3): f()
        torch.cuda.synchronize()
        for setting,indices in [('hot_l0',[0]),('rotate28',list(range(28)))]:
            fs=[functions[name][i] for i in indices]
            def batch_call(fs=fs): return [f() for f in fs]
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): batch_call()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out=batch_call()
            for value in out: value.fill_(float('nan'))
            g.replay();torch.cuda.synchronize()
            for index,value in zip(indices,out):
                cpu=value.cpu().numpy();r=samples[step,index]['y']
                need(cpu.shape==r.shape and np.isfinite(cpu).all() and np.all(np.abs(cpu.astype('float64')-r)<=.002+.002*np.abs(r.astype('float64'))), 'Graph replay numerical check failed')
            # Verify replay consumes CURRENT values at captured addresses, not a
            # correct result left by capture. Alternate is an existing held-out
            # numerical fixture; this is not a performance selection sample.
            alternate=16 if step==1 else 1
            x[0].copy_(torch.from_numpy(samples[alternate,0]['x']).to(x[0].device))
            for value in out:value.fill_(float('nan'))
            g.replay();torch.cuda.synchronize()
            check=out[0].cpu().numpy().astype('float64');reference=samples[alternate,0]['y'].astype('float64')
            need(np.isfinite(check).all() and np.all(np.abs(check-reference)<=.002+.002*np.abs(reference)), 'Graph failed updated-input replay control')
            x[0].copy_(before_x[0]);g.replay();torch.cuda.synchronize()
            graphs[name,setting]=g;outputs[name,setting]=out
        if name!='original': compiler_dump(plans[name][0],folder/'compiler'/name)
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();end.record();end.synchronize()
    measurements=[]
    for setting,indices in [('hot_l0',[0]),('rotate28',list(range(28)))]:
        repeats=100 if setting=='hot_l0' else 10
        for policy in ('graph','eager'):
            for sample in range(30):
                shift=(sample+t['round'])%len(names)
                order=names[shift:]+names[:shift]
                if sample%2:order=list(reversed(order))
                for name in order:
                    if policy=='graph':call=graphs[name,setting].replay
                    else:
                        fs=[functions[name][i] for i in indices]
                        def call(fs=fs):
                            for f in fs:f()
                    begin.record()
                    for _ in range(repeats): call()
                    end.record();end.synchronize()
                    ms=float(begin.elapsed_time(end))
                    need(np.isfinite(ms) and ms>0,'Invalid timing')
                    item={'arm':name,'setting':setting,'policy':policy,'sample':sample,
                        'interval_ms':ms,'replays':repeats,'calls_per_replay':len(indices),
                        'us_per_pipeline':ms*1000/(repeats*len(indices))}
                    measurements.append(item);append(folder/'intervals.jsonl',item)
            print(f'TIMING_SECTION step={step} round={t["round"]} {setting} {policy}',flush=True)
    torch.cuda.synchronize()
    need(all(torch.equal(a,b) for a,b in zip(w,before_w)) and all(torch.equal(a,b) for a,b in zip(x,before_x)), 'Timing changed inputs/weights')
    final_checks=[]
    for name in names:
        for policy in ('eager','graph'):
            if policy=='graph':
                for value in outputs[name,'rotate28']:value.fill_(float('nan'))
                graphs[name,'rotate28'].replay();values=outputs[name,'rotate28']
            else:values=[f() for f in functions[name]]
            torch.cuda.synchronize()
            for layer,value in enumerate(values):
                need(value.dtype==torch.float16,'Post-timing output dtype differs')
                oracle=np.load(run/'numerical'/f'oracle-step{step:02d}-layer{layer:02d}.npz',allow_pickle=False)['y64']
                check=numerical_check(value.cpu().numpy(),samples[step,layer]['y'],oracle)
                record={'arm':name,'policy':policy,'layer':layer,'check':check}
                final_checks.append(record);append(folder/'post-timing-checks.jsonl',record)
                need(check['pass'],'Post-timing full numerical gate failed')
    medians={name:{setting:{policy:statistics.median(v['us_per_pipeline'] for v in measurements if v['arm']==name and v['setting']==setting and v['policy']==policy) for policy in ('graph','eager')} for setting in ('hot_l0','rotate28')} for name in names}
    result={'status':'TIMING_WORKER_PASS','task':t,'samples':measurements,'medians':medians,'setup':plan_info,'post_timing_checks':final_checks,
        'readonly':True,'graph_outputs_checked':True,'graph_nan_reset_and_changed_input_controls':True,'candidate_source_sha256':cfg['candidate_source_sha256'],
        'scope':'FP16 fixed single-token GateUp+activation; separate Graph/Eager; reused outputs versus original allocating APIs; no hardware bandwidth or pure-kernel claim'}
    save(folder/'result.json',result)
    print('PRIMARY_MEDIANS='+str({n:medians[n]['rotate28']['graph'] for n in names}),flush=True)

def run_worker(run,kind,task=None):
    log=run/f'{kind}.log' if task is None else task.parent/'worker.log'
    args=[sys.executable,'-B','-u',str(run/'runner.py'),'--worker',kind,'--run',str(run)]
    if task:args+=['--task',str(task)]
    with log.open('xb') as f:
        p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=PROJECT,timeout=2400)
    need(p.returncode==0,f'{kind} worker failed, evidence: {log}')

def launch_round(run,arms,round_id,step,stage):
    folder=run/stage/f'step{step:02d}-round{round_id}';folder.mkdir(parents=True)
    task=folder/'task.json';save(task,{'arms':arms,'round':round_id,'step':step,'stage':stage})
    print(f'RUN stage={stage} step={step} round={round_id} arms={len(arms)}',flush=True)
    run_worker(run,'timing',task)
    r=read(folder/'result.json')
    print('PRIMARY_MEDIANS='+str({n:r['medians'][n]['rotate28']['graph'] for n in arms}),flush=True)
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-run',type=Path);p.add_argument('--worker',choices=['validate','timing'])
    p.add_argument('--run',type=Path);p.add_argument('--task',type=Path)
    a=p.parse_args()
    if a.worker:
        try:return validate(a.run) if a.worker=='validate' else timing(a.run,a.task)
        except BaseException:
            folder=a.run if a.task is None else a.task.parent
            save(folder/(a.worker+'-failure.json'),{'status':'FAILED','error':traceback.format_exc(),'argv':sys.argv})
            raise
    need(a.fixture_run is not None,'--fixture-run required')
    fixture_run=a.fixture_run.resolve(strict=True);fixture=read(fixture_run/'fixture.json')
    need(fixture['status']=='GATE_UP_FIXTURE_CAPTURE_PASS','Original fixture incomplete')
    parent=PROJECT/'experiments/m08-gate-up-experiments';parent.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='run-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'),dir=parent))
    print('RUN_DIRECTORY='+str(run),flush=True)
    shutil.copyfile(__file__,run/'runner.py');shutil.copyfile(HERE/'50-m08-gate-up-candidates.py',run/'candidates.py')
    mod=load(run/'candidates.py','candidate_declarations')
    plan={'fixture_run':str(fixture_run),'fixture_sha256':sha(fixture_run/'fixture.json'),
        'candidate_source_sha256':sha(run/'candidates.py'),'candidates':mod.candidates(),
        'frozen_files':{n:sha(run/n) for n in ('runner.py','candidates.py')},
        'numerical_rule':{'all_elements_atol':.002,'rtol':.002,'nrmse_multiplier':1.10,'nrmse_floor':1e-6,
            'oracle':'FP64 dot; Y uses FP16-rounded dot then FP64 SiLU/product before final output rounding'},
        'primary':'rotate28_graph_step1_full_GateUp_plus_SwiGLU',
        'timing':{'fresh_process_rounds':3,'interval_samples':30,'hot_replays':100,'rotate28_replays':10},
        'selection_rule':'Best median of paired per-round original/candidate ratios among candidates passing all84 numeric cases; no timing on failed candidates. Freeze once before holdout step16/31 timing.',
        'strong_local_gain_rule':'Median paired latency reduction >=10%, every one of three process rounds reduction >=5%. Engineering threshold, not statistical confidence.',
        'model_rule':'Selected candidate may undergo investigative model check even without strong local gain; exact32 greedy tokens and all32 full logits finite and abs_error<=0.002+0.002*abs(original) required before model timing. No automatic promotion.',
        'author':'coach; independent learner mastery not assessed'}
    save(run/'plan.json',plan)
    result={'status':'FAILED','candidate_source_sha256':plan['candidate_source_sha256'],'fixture_sha256':plan['fixture_sha256'],'plan_sha256':sha(run/'plan.json'),'selected_candidate':None}
    try:
        run_worker(run,'validate')
        n=read(run/'numerical.json');eligible=n['eligible'];result['numerical_eligible']=eligible
        print('NUMERICAL_ELIGIBLE='+str(eligible),flush=True)
        if eligible:
            rounds=[launch_round(run,['original']+eligible,r,1,'selection') for r in (1,2,3)]
            ratios={name:[r['medians']['original']['rotate28']['graph']/r['medians'][name]['rotate28']['graph'] for r in rounds] for name in eligible}
            selected=max(eligible,key=lambda name:statistics.median(ratios[name]))
            rs=ratios[selected];drops=[1-1/v for v in rs]
            selection={'primary':'rotate28_graph_step1','selected_candidate':selected,'ratios_all':ratios,
                'per_round_speedups':rs,'representative_speedup':statistics.median(rs),
                'strong_local_gain':bool(statistics.median(drops)>=.10 and min(drops)>=.05)}
            save(run/'selection.json',selection)
            result.update(selected_candidate=selected,selection=selection,selection_sha256=sha(run/'selection.json'))
            print('FROZEN_SELECTION='+str(selection),flush=True)
            holdouts=[launch_round(run,['original',selected],r,step,'holdout') for step in (16,31) for r in (1,2,3)]
            holdout={'status':'PASS','meaning':'All holdout correctness/timing collection completed; performance acceptance is separate',
                'per_step_ratios':{str(step):[r['medians']['original']['rotate28']['graph']/r['medians'][selected]['rotate28']['graph'] for r in holdouts if r['task']['step']==step] for step in (16,31)}}
            save(run/'holdout.json',holdout);result['holdout']=holdout
        else:
            result['selection']={'strong_local_gain':False,'reason':'No candidate passed frozen numerical gates'}
            result['holdout']={'status':'NOT_RUN'}
        result['status']='GATE_UP_CAMPAIGN_COMPLETE'
    except BaseException:
        result['error']=traceback.format_exc();raise
    finally:
        save(run/'result.json',result)
        files={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()}
        save(run/'artifact-manifest.json',files)
        shared=HERE/'logs/m08-gate-up-experiments'/run.name;shared.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(run,shared)
        print('SHARED_RUN_DIRECTORY='+str(shared),flush=True)
    print('CAMPAIGN_RESULT='+str(result),flush=True)

if __name__=='__main__':main()
