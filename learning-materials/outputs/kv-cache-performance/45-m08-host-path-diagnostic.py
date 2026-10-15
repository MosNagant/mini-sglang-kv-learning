"""Isolate adapter host work, plus real GPU/Graph correctness; coach implementation.

The CPU-only timing replaces ONLY the GPU launcher with a no-op in BOTH adapters.
It measures the wrappers on real CUDA-tensor metadata, not GPU/kernel performance.
Separate GPU calls use the unchanged real kernel and full-backing reference checks.
No profiling samples are used as model timings. Frozen originals are not edited.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
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
BASE = '40-m08-kv-backends.py'
CANDIDATE = '44-m08-kv-cached-metadata.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def worker(run, round_id):
    import numpy as np
    import torch
    plan = json.loads((run/'plan.json').read_text())
    for name, digest in plan['files'].items():
        if sha(run/name) != digest:
            raise RuntimeError('Changed frozen file: '+name)
    out = run/f'round-{round_id}'
    out.mkdir()
    result = {'status':'FAILED', 'round':round_id, 'cases':[]}
    try:
        base = load(run/BASE, 'diag_base')
        candidate = load(run/CANDIDATE, 'diag_candidate')
        sys.path.insert(0, str(run/'source/python'))
        from minisgl.kernel.store import store_cache as original
        def forbidden(*args, **kwargs):
            raise RuntimeError('Unexpected original fallback')
        real = {
            'original':original,
            'old':base.make_store_cache('triton', original_store=forbidden, observe=True),
            'cached':candidate.make_store_cache('triton', original_store=forbidden, observe=True, base_module=base),
        }
        if real['old'].kernel is not real['cached'].kernel:
            raise RuntimeError('Candidate must use identical JIT kernel object')
        # Build a second pair whose GPU dispatch alone is stubbed out.
        real_factory = base._triton_kernel
        triton, _ = real_factory()
        class NoopKernel:
            def __getitem__(self, grid):
                return self.launch
            def launch(self, *args, **kwargs):
                return None
        base._triton_kernel = lambda: (triton, NoopKernel())
        try:
            host = {
                'old':base.make_store_cache('triton', original_store=forbidden, observe=False),
                'cached':candidate.make_store_cache('triton', original_store=forbidden, observe=False, base_module=base),
            }
        finally:
            base._triton_kernel = real_factory
        def exact(a, b):
            return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        for batch in [1, 128]:
            case = {'B':batch, 'D':1024, 'C':513, 'gpu_checks':[], 'host_samples_us':{'old':[], 'cached':[]}}
            qkv = (torch.arange(batch*4096, device='cuda', dtype=torch.int32)%997).to(torch.float16).reshape(batch,4096)
            k, v = qkv[:,2048:3072], qkv[:,3072:4096]
            pool = torch.full((2,513,8,128), -7., device='cuda', dtype=torch.float16)
            indices = torch.arange(batch, device='cuda', dtype=torch.int32)+7
            tensors = (pool[0],pool[1],indices,k,v)
            before_qkv, before_indices = qkv.clone(), indices.clone()
            expected = pool.clone()
            expected[0].view(513,1024)[indices.long()] = k
            expected[1].view(513,1024)[indices.long()] = v
            for arm, call in real.items():
                pool.fill_(-7.)
                call(*tensors)
                torch.cuda.synchronize()
                if not exact(pool,expected) or not exact(qkv,before_qkv) or not exact(indices,before_indices):
                    raise AssertionError(f'{arm} B{batch} Eager mismatch')
                # Capture on a side stream after warming that exact specialization.
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    call(*tensors)
                torch.cuda.current_stream().wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    call(*tensors)
                # Change VALUES at the same capture addresses, including indices.
                qkv.add_(1.)
                indices.add_(1)
                pool.fill_(-7.)
                changed_expected = pool.clone()
                changed_expected[0].view(513,1024)[indices.long()] = k
                changed_expected[1].view(513,1024)[indices.long()] = v
                changed_qkv, changed_indices = qkv.clone(), indices.clone()
                graph.replay()
                torch.cuda.synchronize()
                if not exact(pool,changed_expected) or not exact(qkv,changed_qkv) or not exact(indices,changed_indices):
                    raise AssertionError(f'{arm} B{batch} Graph/value mutation mismatch')
                np.savez_compressed(out/f'B{batch}-{arm}.npz', actual=pool.cpu().numpy(), expected=changed_expected.cpu().numpy(),
                    input_before=changed_qkv.cpu().numpy(), input_after=qkv.cpu().numpy(), indices_before=changed_indices.cpu().numpy(), indices_after=indices.cpu().numpy())
                if arm != 'original':
                    compiled = call.compiled_kernel
                    case.setdefault('kernel_hashes', {})[arm] = {kind: hashlib.sha256(compiled.asm[kind].encode()).hexdigest() for kind in ['ptx','ttgir']}
                    for kind in ['ptx','ttgir']:
                        (out/f'B{batch}-{arm}.{kind}').write_text(compiled.asm[kind], encoding='utf-8')
                # Restore pristine values so every arm has the same workload.
                qkv.copy_(before_qkv)
                indices.copy_(before_indices)
                case['gpu_checks'].append({'arm':arm, 'eager_full_backing':True, 'graph_changed_values_full_backing':True, 'readonly_inputs':True})
            if case['kernel_hashes']['old'] != case['kernel_hashes']['cached']:
                raise AssertionError('GPU compiler artifacts differ in host-only intervention')
            # Check new same-signature objects: no caching of old tensor pointers.
            new_qkv, new_pool, new_indices = qkv.clone(), pool.clone(), indices.clone()
            new_qkv.add_(13.)
            new_indices.add_(3)
            new_pool.fill_(-7.)
            new_args = (new_pool[0],new_pool[1],new_indices,new_qkv[:,2048:3072],new_qkv[:,3072:4096])
            replacement_expected = new_pool.clone()
            replacement_expected[0].view(513,1024)[new_indices.long()] = new_args[3]
            replacement_expected[1].view(513,1024)[new_indices.long()] = new_args[4]
            replacement_input, replacement_indices = new_qkv.clone(), new_indices.clone()
            real['cached'](*new_args)
            torch.cuda.synchronize()
            if not exact(new_pool,replacement_expected) or not exact(new_qkv,replacement_input) or not exact(new_indices,replacement_indices):
                raise AssertionError('Cache hit with replacement tensor objects failed')
            np.savez_compressed(out/f'B{batch}-replacement.npz', actual=new_pool.cpu().numpy(), expected=replacement_expected.cpu().numpy(),
                input_before=replacement_input.cpu().numpy(), input_after=new_qkv.cpu().numpy(), indices_before=replacement_indices.cpu().numpy(), indices_after=new_indices.cpu().numpy())
            case['replacement_objects_checked'] = True
            for call in host.values():
                for _ in range(100): call(*tensors)
            torch.cuda.synchronize()
            unchanged_pool = pool.clone()
            for sample in range(30):
                order = ['old','cached'] if (sample+round_id)%2 else ['cached','old']
                for arm in order:
                    start = time.perf_counter_ns()
                    for _ in range(200): host[arm](*tensors)
                    elapsed = (time.perf_counter_ns()-start)/200/1000
                    case['host_samples_us'][arm].append(elapsed)
            if not exact(pool,unchanged_pool) or not exact(qkv,before_qkv) or not exact(indices,before_indices):
                raise AssertionError('Host-only stub unexpectedly changed device data')
            case['host_medians_us'] = {arm: statistics.median(values) for arm,values in case['host_samples_us'].items()}
            case['host_change_pct'] = (case['host_medians_us']['cached']/case['host_medians_us']['old']-1)*100
            result['cases'].append(case)
            save(out/'result.json',result)
        result['status']='HOST_DIAGNOSTIC_AND_GPU_CORRECTNESS_PASS'
        result['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(0)}
        result['limit']='CPU wrapper timings with no-op launch; not GPU kernel time or model speedup. Graph test is isolated KV, not full-model Graph logits coverage.'
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        save(out/'result.json',result)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--accepted-timing',type=Path)
    ap.add_argument('--worker',type=Path)
    ap.add_argument('--round',type=int)
    args=ap.parse_args()
    if args.worker:
        worker(args.worker,args.round)
        return
    if not args.accepted_timing: ap.error('--accepted-timing required')
    accepted=args.accepted_timing.resolve()
    if json.loads((accepted/'summary.json').read_text())['status']!='MODEL_TIMING_COLLECTION_COMPLETE':
        raise RuntimeError('Expected accepted M08 model timing evidence')
    parent=PROJECT/'experiments/m08-host-diagnostic-results'
    parent.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    print('RUN_DIRECTORY='+str(run),flush=True)
    result={'status':'FAILED','rounds':[]}
    try:
        cfg=json.loads((accepted/'plan.json').read_text())
        shutil.copytree(accepted/'source',run/'source')
        files={}
        for name in [BASE,CANDIDATE,Path(__file__).name]:
            shutil.copyfile(HERE/name,run/name)
            files[name]=sha(run/name)
        for name,digest in cfg['source_manifest'].items():
            if sha(run/'source'/name)!=digest: raise RuntimeError('Frozen source hash mismatch')
            files['source/'+name]=digest
        save(run/'plan.json',{'files':files,'accepted_timing':str(accepted),'scope':'same real-layout KV metadata; B1/128 D1024 C513; CPU no-op dispatch isolation + separate GPU correctness',
            'rounds':3,'host_intervals':30,'calls_per_interval':200,'arms':['old','cached'],
            'decision':'Report all three rounds; no GPU or model performance claim from host-only timing'})
        for rnd in [1,2,3]:
            with (run/f'round-{rnd}.log').open('wb') as log:
                proc=subprocess.run([sys.executable,'-B','-u',str(run/Path(__file__).name),'--worker',str(run),'--round',str(rnd)],stdout=log,stderr=subprocess.STDOUT,timeout=300,cwd=PROJECT)
            if proc.returncode: raise RuntimeError(f'Worker failed; see round-{rnd}.log')
            entry=json.loads((run/f'round-{rnd}/result.json').read_text())
            if entry['status']!='HOST_DIAGNOSTIC_AND_GPU_CORRECTNESS_PASS': raise RuntimeError('Incomplete worker')
            result['rounds'].append(entry)
            print('ROUND '+str(rnd)+' '+json.dumps([{'B':x['B'],'host_medians_us':x['host_medians_us'],'change_pct':x['host_change_pct']} for x in entry['cases']]),flush=True)
        result['status']='HOST_DIAGNOSTIC_COMPLETE'
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        save(run/'summary.json',result)
        shared=HERE/'logs/m08-host-diagnostic-results'/run.name
        shared.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(run,shared)
        print('SHARED_RUN_DIRECTORY='+str(shared),flush=True)

if __name__=='__main__': main()
