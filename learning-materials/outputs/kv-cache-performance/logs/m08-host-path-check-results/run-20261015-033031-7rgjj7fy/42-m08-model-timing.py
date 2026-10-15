"""M08 real-model paired timing, gated on saved 41 correctness evidence.

Coach implementation. Same fixed token workload, original raw store versus
explicit Triton adapter; three fresh process pairs per execution policy.
All samples/outputs retained. No guaranteed speedup or service benchmark claim.
"""
from __future__ import annotations
import argparse
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
import time
import traceback

HERE=Path(__file__).resolve().parent
PROJECT=Path('/home/weltlaw/projects/mini-sglang-kv')

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def worker(run,arm,policy,round_id):
    cfg=json.loads((run/'plan.json').read_text(encoding='utf-8'))
    out=run/f'{policy}-round{round_id}-{arm}'
    out.mkdir()
    result={'status':'FAILED','arm':arm,'policy':policy,'round':round_id,'samples':[]}
    llm=None
    try:
        for name,expected in cfg['source_manifest'].items():
            if digest(run/'source'/name)!=expected:
                raise RuntimeError('Frozen source changed: '+name)
        if digest(run/'adapter.py')!=cfg['adapter_sha256']:
            raise RuntimeError('Frozen adapter changed')
        for f in cfg['model']['weights']:
            stat=(Path(cfg['model']['path'])/f['name']).stat()
            if stat.st_size!=f['bytes'] or stat.st_mtime_ns!=f['mtime_ns']:
                raise RuntimeError('Model weights changed after correctness acceptance')
        for name,expected_sha in cfg['model']['metadata_sha256'].items():
            if digest(Path(cfg['model']['path'])/name)!=expected_sha:
                raise RuntimeError('Model/tokenizer metadata changed: '+name)
        sys.path.insert(0,str(run/'source/python'))
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MINISGL_DISABLE_OVERLAP_SCHEDULING='1')
        import torch
        import minisgl.kernel as kernel_api
        from minisgl.kernel.store import store_cache as original
        from minisgl.llm import LLM
        from minisgl.core import SamplingParams
        adapter=load(run/'adapter.py','m08_timing_adapter')
        def forbidden(*args,**kwargs):
            raise RuntimeError('Triton attempted original fallback')
        observed=adapter.make_store_cache(arm,original_store=original if arm=='original' else forbidden,observe=True)
        measured=original if arm=='original' else adapter.make_store_cache('triton',original_store=forbidden,observe=False)
        kernel_api.store_cache=observed
        raw=[]
        class CountedLLM(LLM):
            def offline_send_result(self,reply):
                raw.extend(int(m.next_token) for m in reply)
                return super().offline_send_result(reply)
        opts=dict(cfg['engine_options'])
        opts['cuda_graph_bs']=[1] if policy=='decode_graph' else []
        llm=CountedLLM(cfg['model']['path'],dtype=torch.float16,**opts)
        if bool(llm.engine.graph_runner.max_graph_bs)!=(policy=='decode_graph'):
            raise RuntimeError('Actual graph configuration mismatch')
        params=SamplingParams(temperature=0.,ignore_eos=True,max_tokens=cfg['max_new_tokens'])
        def generate():
            raw.clear()
            return llm.generate([cfg['input_token_ids']],params)
        expected=cfg['raw_expected_token_ids']
        if len(expected)!=cfg['max_new_tokens']:
            raise RuntimeError('Expected raw step count disagrees with requested workload')
        for _ in range(2):
            generate()
            torch.cuda.synchronize()
            if raw!=expected:
                raise RuntimeError('Warmup output differs from accepted original')
        if observed.launch_count<=0:
            raise RuntimeError('Selected backend never reached a real launch')
        result['initial_backend_observation']={'launch_count':observed.launch_count,'last_launch':observed.last_launch,
             'scope':'Initialization/capture/warmup Python submissions only; not Graph replay node counts'}
        if arm=='triton':
            compiled=observed.compiled_kernel
            if compiled is None:
                raise RuntimeError('Actual Triton compiled specialization missing')
            result['triton_kernel_name']=compiled.name
            for kind in ['ptx','ttgir']:
                (out/f'warmup-kernel.{kind}').write_text(compiled.asm[kind],encoding='utf-8')
        # Initialization/capture uses the same device kernel as measured. The
        # observer is removed before measuring Python-dispatched prefill/Eager.
        kernel_api.store_cache=measured
        result['environment']={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
             'cuda':torch.version.cuda,'model_dtype':'torch.float16','engine_options':opts,
              'model_stream':int(llm.engine.stream.cuda_stream)}
        result['environment']['env']={k:os.environ.get(k) for k in
            ['CUDA_VISIBLE_DEVICES','CUDA_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','MINISGL_DISABLE_OVERLAP_SCHEDULING']}
        result['environment']['packages']={name:importlib.metadata.version(name) for name in
            ['torch','triton','flashinfer-python','transformers','safetensors','apache-tvm-ffi']}
        result['environment']['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.used',
            '--format=csv,noheader'],capture_output=True,text=True,timeout=30).stdout
        if result['environment']['packages']!=cfg['timing_packages']:
            raise RuntimeError('Environment package versions changed during collection')
        torch.cuda.reset_peak_memory_stats()
        for sample in range(5):
            torch.cuda.synchronize()
            t0=time.perf_counter()
            output=generate()
            torch.cuda.synchronize()
            seconds=time.perf_counter()-t0
            # Output inspection and disk writes are outside the timed interval.
            entry={'sample':sample,'seconds':seconds,'raw_generated_token_ids':list(raw),
                   'raw_generated_steps':len(raw),'returned_tokens':len(output[0]['token_ids']),
                   'raw_output_tokens_per_second':len(raw)/seconds,'output':output}
            result['samples'].append(entry)
            save(out/'result.json',result)
            if raw!=expected or seconds<=0:
                raise RuntimeError('Timed sample output mismatch or invalid duration')
        result['median_seconds']=statistics.median(x['seconds'] for x in result['samples'])
        result['raw_steps_per_second_at_median']=len(expected)/result['median_seconds']
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['status']='MODEL_TIMING_WORKER_COMPLETE'
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException:
                result['shutdown_error']=traceback.format_exc()
                result['status']='FAILED'
        save(out/'result.json',result)
    if result['status']!='MODEL_TIMING_WORKER_COMPLETE':
        raise RuntimeError('Worker cleanup failed')
    print(f"COLLECTED {policy} round={round_id} {arm} median_ms={result['median_seconds']*1000:.6f} raw_tokens_per_second={result['raw_steps_per_second_at_median']:.3f}",flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--accepted-check',type=Path)
    ap.add_argument('--worker-run',type=Path)
    ap.add_argument('--arm',choices=['original','triton'])
    ap.add_argument('--policy',choices=['eager','decode_graph'])
    ap.add_argument('--round',type=int)
    args=ap.parse_args()
    if args.worker_run:
        worker(args.worker_run,args.arm,args.policy,args.round)
        return
    if not args.accepted_check:
        ap.error('--accepted-check is required')
    accepted=args.accepted_check.resolve()
    check=json.loads((accepted/'result.json').read_text(encoding='utf-8'))
    request=json.loads((accepted/'request.json').read_text(encoding='utf-8'))
    if check['status']!='MODEL_AB_CORRECTNESS_PASS':
        raise RuntimeError('41 real model correctness has not passed')
    if digest(HERE/'40-m08-kv-backends.py')!=request['adapter_sha256']:
        raise RuntimeError('Candidate changed since correctness; do not time')
    if digest(HERE/'39-m08-model-smoke.py')!=request['helper_sha256']:
        raise RuntimeError('Inspection helper changed since 41 correctness')
    helper=load(HERE/'39-m08-model-smoke.py','m08_smoke_helpers')
    # A fresh full model hash is obtained once, not in each timed process.
    model=helper.inspect_model(Path(request['model']['path']),request['model']['revision'])
    if [(f['name'],f['sha256']) for f in model['weights']] != [(f['name'],f['sha256']) for f in request['model']['weights']]:
        raise RuntimeError('Model weights differ from accepted correctness run')
    if model['metadata_sha256']!=request['model']['metadata_sha256']:
        raise RuntimeError('Model/tokenizer metadata differ from accepted correctness run')
    parent=PROJECT/'experiments/m08-model-timing-results'
    parent.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    print('RUN_DIRECTORY='+str(run),flush=True)
    aggregate={'status':'RUNNING','workers':[]}
    try:
        sources=helper.snapshot_source(PROJECT,run)
        # helper snapshots itself; retain that exact helper separately and then
        # freeze this timing controller as the worker entry.
        (run/'runner.py').rename(run/'smoke-helper.py')
        shutil.copyfile(__file__,run/'runner.py')
        shutil.copyfile(HERE/'40-m08-kv-backends.py',run/'adapter.py')
        accepted_manifest=json.loads((accepted/'source-manifest.json').read_text(encoding='utf-8'))
        if sources!=accepted_manifest:
            raise RuntimeError('Upstream source changed since 41 correctness')
        order=[]
        for rnd in [1,2,3]:
            policies=['decode_graph','eager'] if rnd%2 else ['eager','decode_graph']
            arms=['original','triton'] if rnd%2 else ['triton','original']
            order += [{'round':rnd,'policy':p,'arm':a} for p in policies for a in arms]
        plan={'model':model,'input_token_ids':request['input_token_ids'],
              'max_new_tokens':request['max_new_tokens'],'raw_expected_token_ids':check['raw_expected_token_ids'],
              'engine_options':request['engine_options'],'adapter_sha256':request['adapter_sha256'],
              'source_manifest':sources,'accepted_check':str(accepted),'accepted_result_sha256':digest(accepted/'result.json'),
              'timing_packages':{name:importlib.metadata.version(name) for name in
                  ['torch','triton','flashinfer-python','transformers','safetensors','apache-tvm-ffi']},
              'order':order,'warmups_per_worker':2,'samples_per_worker':5,
              'measurement':'perf_counter around generate + GPU completion; prepared token IDs; scheduler/sampling/D2H/detokenize included; no network/model loading/JIT/warmup',
              'baseline':'raw upstream original store; candidate observe=False metadata adapter; Graph/Eager separate',
              'signal_rule':'All 3 paired process medians at least 3% faster is a scoped engineering signal, not statistical significance; keep all outcomes',
              'scope':'1 request, 128 input tokens, 32 generation steps, Qwen3-0.6B FP16 TP1; decode Graph only; prefill always Eager; no service/multi-request claim'}
        if plan['timing_packages']!=request['packages']:
            raise RuntimeError('Package environment changed since 41 correctness')
        assert len(plan['input_token_ids'])==128 and plan['max_new_tokens']==32
        save(run/'accepted-request.json',request)
        save(run/'accepted-result.json',check)
        save(run/'plan.json',plan)
        for item in order:
            label=f"{item['policy']}-round{item['round']}-{item['arm']}"
            print('RUN '+label,flush=True)
            with (run/(label+'.log')).open('wb') as log:
                proc=subprocess.run([sys.executable,'-B','-u',str(run/'runner.py'),'--worker-run',str(run),
                       '--arm',item['arm'],'--policy',item['policy'],'--round',str(item['round'])],
                       cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            worker_result=run/label/'result.json'
            if not worker_result.exists():
                raise RuntimeError(f'Worker exited {proc.returncode} without result; inspect retained {label}.log')
            entry=json.loads(worker_result.read_text(encoding='utf-8'))
            aggregate['workers'].append(entry)
            save(run/'summary.json',aggregate)
            if proc.returncode or entry['status']!='MODEL_TIMING_WORKER_COMPLETE':
                raise RuntimeError('Worker failed; retained '+label+'.log')
            print(f"COLLECTED {label} median_ms={entry['median_seconds']*1000:.6f}",flush=True)
        aggregate['comparisons']={}
        for policy in ['eager','decode_graph']:
            pairs=[]
            for rnd in [1,2,3]:
                entries={x['arm']:x for x in aggregate['workers'] if x['policy']==policy and x['round']==rnd}
                a,b=[entries[k]['median_seconds'] for k in ['original','triton']]
                pairs.append({'round':rnd,'original_ms':a*1000,'triton_ms':b*1000,'original_over_triton':a/b,'candidate_time_change_pct':(b/a-1)*100})
            aggregate['comparisons'][policy]={'pairs':pairs,'all_rounds_at_least_3pct_faster':all(x['candidate_time_change_pct']<=-3 for x in pairs)}
        aggregate['status']='MODEL_TIMING_COLLECTION_COMPLETE'
        aggregate['limit']=plan['scope']+'; no automatic performance advantage or model quality claim'
    except BaseException:
        aggregate['status']='FAILED'
        aggregate['error']=traceback.format_exc()
        raise
    finally:
        save(run/'summary.json',aggregate)
        shared=HERE/'logs/m08-model-timing-results'/run.name
        try:
            shared.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(run,shared)
            print('SHARED_RUN_DIRECTORY='+str(shared),flush=True)
        except Exception:
            print('SHARED_COPY_FAILED; local evidence retained: '+str(run),flush=True)
            if aggregate['status']!='FAILED':
                raise
    print(json.dumps(aggregate['comparisons'],indent=2),flush=True)

if __name__=='__main__':
    try:
        main()
    except BaseException:
        # Also retain failures occurring before a model run can be created.
        failure=traceback.format_exc()
        try:
            failure_parent=HERE/'logs/m08-model-timing-errors'
            failure_parent.mkdir(parents=True,exist_ok=True)
            failure_run=Path(tempfile.mkdtemp(prefix='error-',dir=failure_parent))
            save(failure_run/'error.json',{'argv':sys.argv,'error':failure,'runner_sha256':digest(__file__)})
        except Exception:
            pass
        raise
