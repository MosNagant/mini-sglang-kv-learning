"""M08 minimal dependency install / pinned public model download, explicit modes."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['dependency-plan', 'dependency-install', 'model'])
a = ap.parse_args()
root = Path(__file__).resolve().parent
parent = root / 'logs/m08-setup-results'
parent.mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix=a.mode+'-', dir=parent))
shutil.copyfile(__file__, run/'setup.py')
result = {'mode': a.mode, 'python': sys.executable, 'status': 'RUNNING'}
print('RUN_DIRECTORY='+str(run), flush=True)
def packages():
    return {d.metadata['Name']: d.version for d in md.distributions()}
def fetch_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)
try:
    result['packages_before'] = packages()
    if a.mode.startswith('dependency-'):
        constraints = run/'constraints.txt'
        constraints.write_text(''.join(f'{k}=={v}\n' for k,v in sorted(result['packages_before'].items())), encoding='utf-8')
        args = [shutil.which('uv'), 'pip', 'install', '--python', sys.executable, '-c', str(constraints),
                'flashinfer-python==0.5.3', 'nvidia-cutlass-dsl==4.2.1',
                'nvidia-cudnn-frontend==1.13.0', 'cuda-bindings==12.9.4']
        if a.mode == 'dependency-plan':
            args.append('--dry-run')
        result['command'] = args
        with (run/'install.log').open('w', encoding='utf-8') as output:
            p = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, timeout=600)
        print((run/'install.log').read_text(encoding='utf-8'), flush=True)
        result['returncode'] = p.returncode
        result['packages_after'] = packages()
        result['changed_existing'] = {k: [v, result['packages_after'].get(k)] for k,v in result['packages_before'].items() if result['packages_after'].get(k) != v}
        if p.returncode or result['changed_existing']:
            raise RuntimeError('Dependency operation failed or changed an existing package; review log')
    else:
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        model_id = 'Qwen/Qwen3-0.6B'
        info = fetch_json('https://huggingface.co/api/models/'+model_id+'?blobs=true')
        revision = info['sha']
        (run/'huggingface-metadata.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
        from huggingface_hub import snapshot_download
        destination = Path.home()/'models'/'Qwen3-0.6B'/revision
        print('MODEL_REVISION='+revision, flush=True)
        print('MODEL_DESTINATION='+str(destination), flush=True)
        path = snapshot_download(model_id, revision=revision, local_dir=destination,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'], max_workers=4)
        files=[]
        for f in sorted(destination.iterdir()):
            if not f.is_file():
                continue
            digest=hashlib.sha256()
            with f.open('rb') as fp:
                for block in iter(lambda: fp.read(8*1024*1024), b''):
                    digest.update(block)
            entry={'name':f.name,'bytes':f.stat().st_size,'sha256':digest.hexdigest()}
            remote=next((s for s in info['siblings'] if s['rfilename']==f.name), {})
            expected=remote.get('lfs',{}).get('sha256')
            if expected and expected!=entry['sha256']:
                raise RuntimeError('Model LFS hash mismatch: '+f.name)
            entry['lfs_sha256_verified']=bool(expected)
            files.append(entry)
        result.update(model_id=model_id, revision=revision, model_path=str(destination), files=files)
        (destination/'m08-model-manifest.json').write_text(json.dumps({k:result[k] for k in ['model_id','revision','model_path','files']}, indent=2)+'\n', encoding='utf-8')
        shutil.copyfile(destination/'config.json', run/'config.json')
    result['status']='COMPLETE'
except Exception:
    result['status']='FAILED'
    result['error']=traceback.format_exc()
    print(result['error'], flush=True)
finally:
    (run/'result.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    print('SETUP_STATUS='+result['status']+' SAVED='+str(run/'result.json'),flush=True)
if result['status']!='COMPLETE':
    raise SystemExit(1)
