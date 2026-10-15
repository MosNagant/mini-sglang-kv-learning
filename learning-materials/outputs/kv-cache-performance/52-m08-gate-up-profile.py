"""Bounded Nsight Compute diagnostic for frozen 49 original/selected pipelines.

No installation, permission changes, clock locking or CUDA Graph/range replay.
Query actual metric availability first; stop at the first failed profile. Kernel
replay with cache-control none can perturb cache state: counters describe this
profile, not the uninstrumented 49 Graph benchmark or a measured cache hit rate.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/weltlaw/projects/mini-sglang-kv')
METRICS = ['dram__bytes_read.sum', 'lts__t_bytes.sum',
           'sm__warps_active.avg.pct_of_peak_sustained_active', 'gpu__time_duration.sum']
RANGE = 'M08_GATE_UP_PROFILE'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def need(ok, message):
    if not ok: raise RuntimeError(message)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module;spec.loader.exec_module(module)
    return module


def classify(text, returncode):
    if any(line.startswith('==ERROR== ERR_NVGPUCTRPERM') for line in text.splitlines()):
        return 'GPU_COUNTER_PERMISSION_DENIED'
    if 'Profiling failed because a driver resource was unavailable' in text:
        return 'DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE'
    return 'COMMAND_PASS' if returncode == 0 else 'FAILED_EXIT_' + str(returncode)


def command(args, folder, label, timeout=180):
    save(folder/(label+'-command.json'), args)
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           cwd=PROJECT, timeout=timeout)
        text, code = p.stdout, p.returncode
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout or ''
        if isinstance(text, bytes): text = text.decode(errors='replace')
        text += '\nBOUNDED_TIMEOUT\n';code = 124
    (folder/(label+'.log')).write_text(text, encoding='utf-8')
    return {'returncode': code, 'classification': classify(text, code), 'log': label+'.log'}, text


def worker(run, arm, setting):
    cfg = read(run/'plan.json')
    for filename, digest in cfg['frozen_files'].items(): need(sha(run/filename)==digest, 'Frozen source changed')
    micro_run = Path(cfg['micro_run'])
    need(sha(micro_run/'result.json')==cfg['micro_result_sha256'], 'Accepted micro result changed')
    need(sha(micro_run/'plan.json')==cfg['micro_plan_sha256'], 'Accepted micro plan changed')
    micro = load(run/'micro_runner.py', 'frozen_micro_profile_helper')
    accepted, candidate, weights, samples = micro.setup(micro_run)
    need(accepted['candidate_source_sha256']==cfg['candidate_source_sha256'], 'Candidate changed')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from flashinfer import silu_and_mul
    directory = run/(arm+'-'+setting)
    ids = [0] if setting=='hot_l0' else list(range(28))
    ws = [torch.from_numpy(weights[i].copy()).cuda() for i in ids]
    xs = [torch.from_numpy(samples[1,i]['x'].copy()).cuda() for i in ids]
    before_w, before_x = [v.clone() for v in ws], [v.clone() for v in xs]
    plans = []
    if arm=='original':
        calls = [(lambda x=x,w=w:silu_and_mul(F.linear(x,w))) for x,w in zip(xs,ws)]
        kernels_per_pipeline = 2
    else:
        plans = [candidate.prepare(cfg['selected_candidate'], x, w) for x,w in zip(xs,ws)]
        calls = [(lambda p=p:silu_and_mul(p['call']())) if p['kind']=='gemv' else p['call'] for p in plans]
        kernels_per_pipeline = plans[0]['kernel_launches_per_call'] + int(plans[0]['kind']=='gemv')
    # Compile/setup and ten complete warmup sequences are outside the NVTX scope.
    for _ in range(10):
        for call in calls: call()
    torch.cuda.synchronize()
    outputs = []
    torch.cuda.nvtx.range_push(RANGE)
    try:
        for layer,call in zip(ids,calls):
            torch.cuda.nvtx.mark(f'{arm}:{setting}:layer{layer}')
            outputs.append(call())
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    checks=[]
    for layer,value in zip(ids,outputs):
        need(value.dtype==torch.float16, 'Output dtype changed')
        with np.load(micro_run/'numerical'/f'oracle-step01-layer{layer:02d}.npz',allow_pickle=False) as data:
            check=micro.numerical_check(value.cpu().numpy(),samples[1,layer]['y'],data['y64'])
        checks.append({'layer':layer,'check':check});need(check['pass'], 'Profile output numerical check failed')
    need(all(torch.equal(a,b) for a,b in zip(ws,before_w)) and all(torch.equal(a,b) for a,b in zip(xs,before_x)), 'Profile changed inputs')
    save(directory/'workload.json', {'status':'PROFILE_WORKLOAD_PASS','arm':arm,'setting':setting,
        'layers':ids,'warmup_sequences':10,'profile_sequences':1,'expected_kernel_count':len(ids)*kernels_per_pipeline,
        'checks':checks,'readonly':True,'selected_candidate':cfg['selected_candidate'],
        'limit':'Eager diagnostic launches; NCU may replay each kernel. Not Graph timing or an unperturbed cache experiment.'})


def parse_raw(text):
    lines=text.splitlines()
    start=next((i for i,line in enumerate(lines)
                if {'Metric Name','Kernel Name','Metric Value'} <= set(next(csv.reader([line])))), None)
    need(start is not None, 'NCU raw CSV header missing')
    records=[]
    for row in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        if row.get('Metric Name') not in METRICS: continue
        raw=row.get('Metric Value','')
        value=float(raw.replace(',','').strip())
        need(math.isfinite(value) and value>=0, 'Missing or invalid hardware metric value')
        records.append({'id':row.get('ID'), 'kernel':row.get('Kernel Name'),
                        'metric':row['Metric Name'], 'unit':row.get('Metric Unit'), 'value':value, 'raw':raw})
    groups={}
    for row in records: groups.setdefault(row['id'],set()).add(row['metric'])
    need(groups and all(set(METRICS)<=values for values in groups.values()), 'Incomplete per-kernel metric set')
    return {'records':records,'kernel_count':len(groups)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--micro-run',type=Path)
    parser.add_argument('--query-only',action='store_true')
    parser.add_argument('--worker-run',type=Path)
    parser.add_argument('--arm',choices=('original','candidate'))
    parser.add_argument('--setting',choices=('hot_l0','rotate28'))
    args=parser.parse_args()
    if args.worker_run: return worker(args.worker_run,args.arm,args.setting)
    need(args.micro_run is not None,'--micro-run required')
    micro_run=args.micro_run.resolve(strict=True);accepted=read(micro_run/'result.json');old_plan=read(micro_run/'plan.json')
    need(accepted['status']=='GATE_UP_CAMPAIGN_COMPLETE' and accepted['selected_candidate'] in accepted['numerical_eligible'],'No accepted selected candidate')
    need(sha(micro_run/'plan.json')==accepted['plan_sha256'],'Accepted plan changed')
    need(sha(micro_run/'candidates.py')==accepted['candidate_source_sha256'],'Accepted candidate source changed')
    for name,digest in old_plan['frozen_files'].items(): need(sha(micro_run/name)==digest,'Micro source changed: '+name)
    parent=PROJECT/'experiments/m08-gate-up-profile';parent.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='run-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'),dir=parent))
    print('RUN_DIRECTORY='+str(run),flush=True)
    result={'status':'FAILED','gpu_counter_claim':False,'profiles':[]}
    try:
        shutil.copyfile(__file__,run/'runner.py');shutil.copyfile(micro_run/'runner.py',run/'micro_runner.py')
        shutil.copyfile(HERE/'52-m08-gate-up-profile.sh',run/'52-m08-gate-up-profile.sh')
        cfg={'micro_run':str(micro_run),'micro_result_sha256':sha(micro_run/'result.json'),
             'micro_plan_sha256':sha(micro_run/'plan.json'),'selected_candidate':accepted['selected_candidate'],
             'candidate_source_sha256':accepted['candidate_source_sha256'],'metrics':METRICS,
             'frozen_files':{name:sha(run/name) for name in ('runner.py','micro_runner.py','52-m08-gate-up-profile.sh')},
             'policy':'kernel replay; cache-control none; clock-control none; same fixed step1 fixtures; 10 warmups outside NVTX; one profiled sequence',
             'limit':'Profile cache state can differ across replay passes. No hit-rate, bandwidth or cause claim without interpreting actual metric units/replay overhead.',
             'documentation':'https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html'}
        save(run/'plan.json',cfg)
        ncu=shutil.which('ncu')
        if ncu is None and Path('/usr/local/cuda-12.8/bin/ncu').is_file(): ncu='/usr/local/cuda-12.8/bin/ncu'
        if ncu is None: result['status']='NCU_NOT_ON_PATH';return
        version,_=command([ncu,'--version'],run,'ncu-version',30);need(version['returncode']==0,'NCU version query failed')
        query,text=command([ncu,'--query-metrics','--query-metrics-mode','suffix','--metrics',','.join(METRICS),'--csv'],run,'metric-query',60)
        result['metric_query']=query
        if query['returncode']!=0 or not all(metric in text for metric in METRICS):
            result['status']='REQUESTED_METRICS_UNAVAILABLE';return
        result['metric_names_confirmed']=METRICS
        if args.query_only: result['status']='METRICS_QUERY_ONLY_NO_PROFILE';return
        for arm,setting in [('original','hot_l0'),('candidate','hot_l0'),('original','rotate28'),('candidate','rotate28')]:
            folder=run/(arm+'-'+setting);folder.mkdir()
            kernel_filter='regex:gemv|act_and_mul_kernel' if arm=='original' else 'regex:_gemv_|_swiglu_fused|act_and_mul_kernel'
            cmd=[ncu,'--metrics',','.join(METRICS),'--replay-mode','kernel','--cache-control','none','--clock-control','none',
                 '--nvtx','--nvtx-include',RANGE+'/', '--kernel-name-base','function','--kernel-name',kernel_filter,
                 '--export',str(folder/'profile'),sys.executable,'-B','-u',str(run/'runner.py'),
                 '--worker-run',str(run),'--arm',arm,'--setting',setting]
            status,text=command(cmd,folder,'profile',180)
            record={'arm':arm,'setting':setting,**status};result['profiles'].append(record)
            if status['returncode']!=0:
                result['status']=status['classification'];return
            need((folder/'profile.ncu-rep').is_file() and (folder/'workload.json').is_file(),'Profile or workload evidence missing')
            exported,text=command([ncu,'--import',str(folder/'profile.ncu-rep'),'--page','raw','--csv'],folder,'metrics',60)
            need(exported['returncode']==0,'NCU report export failed')
            parsed=parse_raw(text);workload=read(folder/'workload.json')
            need(parsed['kernel_count']==workload['expected_kernel_count'],'NCU kernel filter did not collect the complete expected pipeline')
            save(folder/'metrics.json',parsed);record['kernel_count']=parsed['kernel_count']
        result['status']='COUNTERS_COLLECTED_REVIEW_REQUIRED';result['gpu_counter_claim']=True
        result['limit']='Only instrumented per-kernel metrics; no performance-cause, Graph-memory-traffic, cache-hit-rate or bandwidth conclusion.'
    except BaseException:
        result['error']=traceback.format_exc();raise
    finally:
        save(run/'result.json',result)
        shared=HERE/'logs/m08-gate-up-profile'/run.name;shared.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(run,shared);print('SHARED_RUN_DIRECTORY='+str(shared),flush=True)
        print('PROFILE_STATUS='+result['status'],flush=True)


if __name__=='__main__': main()
