"""Separate original-model Nsight diagnostic; never a timing-suite sample."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback

HERE=Path(__file__).resolve().parent
PROJECT=Path('/home/weltlaw/projects/mini-sglang-kv')
NSYS=Path('/opt/nvidia/nsight-systems/2025.1.3/bin/nsys')

def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def child(run):
    cfg=json.loads((run/'plan.json').read_text(encoding='utf-8'))
    for name,expected in cfg['source_manifest'].items():
        if sha(run/'source'/name)!=expected:
            raise RuntimeError('Profile source changed: '+name)
    for name,expected in cfg['model']['metadata_sha256'].items():
        if sha(Path(cfg['model']['path'])/name)!=expected:
            raise RuntimeError('Profile model metadata changed: '+name)
    for f in cfg['model']['weights']:
        s=(Path(cfg['model']['path'])/f['name']).stat()
        if (s.st_size,s.st_mtime_ns)!=(f['bytes'],f['mtime_ns']):
            raise RuntimeError('Profile checkpoint changed')
    sys.path.insert(0,str(run/'source/python'))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MINISGL_DISABLE_OVERLAP_SCHEDULING='1')
    import torch
    from minisgl.llm import LLM
    from minisgl.core import SamplingParams
    import minisgl.kernel as kernel_api
    from minisgl.kernel.store import store_cache
    if kernel_api.store_cache is not store_cache:
        raise RuntimeError('Original profile entry replaced')
    raw=[]
    class CountedLLM(LLM):
        def offline_send_result(self,reply):
            raw.extend(int(m.next_token) for m in reply)
            return super().offline_send_result(reply)
    opts=dict(cfg['engine_options'],cuda_graph_bs=[1])
    model=None
    result={'status':'FAILED','policy':'decode_graph','backend':'original','not_timing_sample':True}
    try:
        model=CountedLLM(cfg['model']['path'],dtype=torch.float16,**opts)
        sp=SamplingParams(temperature=0.,ignore_eos=True,max_tokens=cfg['max_new_tokens'])
        for _ in range(2):
            raw.clear()
            model.generate([cfg['input_token_ids']],sp)
            torch.cuda.synchronize()
            if raw!=cfg['raw_expected_token_ids']:
                raise RuntimeError('Profile warmup output mismatch')
        raw.clear()
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('M08_ORIGINAL_GENERATE')
        output=model.generate([cfg['input_token_ids']],sp)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        if raw!=cfg['raw_expected_token_ids']:
            raise RuntimeError('Profile output mismatch')
        result.update(status='PROFILE_WORKLOAD_CHECKED',raw_generated_token_ids=list(raw),output=output,
                      expected_kv_calls=28*len(raw),engine_options=opts,
                      packages={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in cfg['timing_packages']})
        if result['packages']!=cfg['timing_packages']:
            raise RuntimeError('Profile package environment changed')
    except BaseException:
        result['status']='FAILED'
        result['error']=traceback.format_exc()
        raise
    finally:
        if model is not None:
            model.shutdown()
        save(run/'workload.json',result)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--timing-run',type=Path)
    ap.add_argument('--child',type=Path)
    a=ap.parse_args()
    if a.child:
        child(a.child)
        return
    if not a.timing_run:
        ap.error('--timing-run required')
    timing=a.timing_run.resolve()
    summary=json.loads((timing/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='MODEL_TIMING_COLLECTION_COMPLETE':
        raise RuntimeError('Use completed unprofiled model timing evidence')
    cfg=json.loads((timing/'plan.json').read_text(encoding='utf-8'))
    parent=PROJECT/'experiments/m08-model-profile-results'
    parent.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    print('RUN_DIRECTORY='+str(run),flush=True)
    result={'status':'RUNNING','timing_run':str(timing),'policy':'decode_graph','backend':'original'}
    try:
        shutil.copytree(timing/'source',run/'source')
        shutil.copyfile(__file__,run/'profile.py')
        save(run/'plan.json',cfg)
        config=Path.home()/'.config/NVIDIA Corporation/nsys-config.ini'
        if config.exists():
            shutil.copyfile(config,run/'nsys-config.ini')
        prefix=run/'original-model'
        command=[str(NSYS),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                 '--cuda-event-trace=false','--cuda-graph-trace=node','--capture-range=cudaProfilerApi',
                 '--capture-range-end=stop','--output='+str(prefix),sys.executable,'-B','-u',str(run/'profile.py'),'--child',str(run)]
        result['command']=command
        with (run/'nsys-profile.log').open('wb') as log:
            p=subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        result['profile_returncode']=p.returncode
        if p.returncode:
            raise RuntimeError('Nsight profile failed; log preserved')
        work=json.loads((run/'workload.json').read_text(encoding='utf-8'))
        if work['status']!='PROFILE_WORKLOAD_CHECKED':
            raise RuntimeError('Profile workload failed')
        with (run/'nsys-kernels.csv').open('wb') as log:
            p=subprocess.run([str(NSYS),'stats','--report','cuda_gpu_kern_sum','--format','csv',str(prefix.with_suffix('.nsys-rep'))],
                             stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if p.returncode:
            raise RuntimeError('Nsight stats failed; raw report preserved')
        db=sqlite3.connect(prefix.with_suffix('.sqlite'))
        strings=dict(db.execute('SELECT id,value FROM StringIds'))
        fields=[r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')]
        namecol='demangledName' if 'demangledName' in fields else 'shortName'
        totals={}
        for start,end,nameid in db.execute(f'SELECT start,end,{namecol} FROM CUPTI_ACTIVITY_KIND_KERNEL'):
            name=strings.get(nameid,str(nameid))
            data=totals.setdefault(name,{'count':0,'total_ns':0})
            data['count']+=1
            data['total_ns']+=end-start
        db.close()
        kv={name:data for name,data in totals.items() if 'store_kv_cache<' in name}
        count=sum(x['count'] for x in kv.values())
        if count!=work['expected_kv_calls']:
            raise RuntimeError(f'KV trace count {count} differs from expected {work["expected_kv_calls"]}; inspect capture')
        all_ns=sum(x['total_ns'] for x in totals.values())
        kv_ns=sum(x['total_ns'] for x in kv.values())
        result.update(status='REAL_MODEL_KV_SHARE_RECORDED',kv_kernels=kv,kv_count=count,
                      kv_total_ns=kv_ns,all_kernel_total_ns=all_ns,kv_kernel_duration_share_pct=100*kv_ns/all_ns,
                      kernels=totals,
                      limit='Separate instrumented single workload; summed GPU kernel durations, not wall-clock critical-path fraction, bandwidth, confidence interval or pure unprofiled kernel timing. Graph node tracing and WSL timestamp settings affect accuracy.')
    except BaseException:
        result['status']='FAILED'
        result['error']=traceback.format_exc()
        raise
    finally:
        save(run/'result.json',result)
        shared=HERE/'logs/m08-model-profile-results'/run.name
        try:
            shared.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(run,shared)
            print('SHARED_RUN_DIRECTORY='+str(shared),flush=True)
        except Exception:
            print('Local profile evidence retained: '+str(run),flush=True)
            if result['status']!='FAILED':
                raise
    print(json.dumps({k:v for k,v in result.items() if k not in ['kernels','command']},indent=2),flush=True)

if __name__=='__main__':
    main()
