"""Read-only M08 prerequisite probe. No package install or model execution."""
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

PROJECT = Path('/home/weltlaw/projects/mini-sglang-kv')
SHARED = Path(__file__).resolve().parent / 'logs/m08-preflight-results'
SHARED.mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix='run-', dir=SHARED))
shutil.copyfile(__file__, run / 'preflight.py')
result = {'scope': 'actual environment/dependency/model-cache preflight; no model run', 'python': sys.executable}
print('RUN_DIRECTORY=' + str(run), flush=True)

def command(args, timeout=40):
    p = subprocess.run(args, cwd=PROJECT, capture_output=True, text=True, timeout=timeout)
    return {'args': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

try:
    result['commit'] = command(['git', 'rev-parse', 'HEAD'])
    result['tracked_status'] = command(['git', 'status', '--porcelain', '--untracked-files=no'])
    result['packages'] = {}
    for name in ['torch', 'triton', 'transformers', 'accelerate', 'modelscope', 'huggingface-hub', 'safetensors', 'flashinfer-python', 'sgl-kernel', 'apache-tvm-ffi', 'quack-kernels', 'pyzmq', 'msgpack', 'fastapi', 'uvicorn', 'openai']:
        try:
            result['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result['packages'][name] = None
    import torch
    result['gpu'] = {'name': torch.cuda.get_device_name(0), 'torch_cuda': torch.version.cuda, 'free_total_bytes': list(torch.cuda.mem_get_info()), 'capability': list(torch.cuda.get_device_capability(0))}
    result['nvidia_smi'] = command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv,noheader'])
    result['engine_import'] = command([sys.executable, '-B', '-c', 'from minisgl.engine import Engine, EngineConfig; print("ENGINE_IMPORT_OK")'])
    roots = [Path.home() / '.cache/huggingface/hub']
    if os.environ.get('HF_HOME'):
        roots.append(Path(os.environ['HF_HOME']) / 'hub')
    if os.environ.get('HF_HUB_CACHE'):
        roots.append(Path(os.environ['HF_HUB_CACHE']))
    result['model_cache'] = []
    for root in dict.fromkeys(roots):
        model_root = root / 'models--Qwen--Qwen3-0.6B/snapshots'
        if model_root.is_dir():
            for snapshot in model_root.iterdir():
                if snapshot.is_dir():
                    result['model_cache'].append({'path': str(snapshot), 'files': [{'name': p.name, 'bytes': p.stat().st_size} for p in snapshot.iterdir() if p.is_file()]})
    result['offline_flags'] = {k: os.environ.get(k) for k in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']}
    result['status'] = 'PREFLIGHT_RECORDED'
except Exception:
    result['status'] = 'PREFLIGHT_ERROR'
    result['error'] = traceback.format_exc()
finally:
    (run / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print('SAVED=' + str(run / 'result.json'), flush=True)
if result['status'] != 'PREFLIGHT_RECORDED':
    raise SystemExit(1)
