"""CPU controls for53/54. Fake GPU interfaces only; no device execution."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

base = Path(__file__).resolve().parents[1]


def load(path, name):
    ast.parse(path.read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load(base/'53-m08-stable-runtime.py', 'runtime53_cpu')
runner = load(base/'54-verify-stable-runtime.py', 'runner54_cpu')
checks = []


def rejected(call, fragment):
    try:
        call()
    except (ValueError,RuntimeError) as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError('Expected rejection: '+fragment)


runtime.validate_inputs(list(range(128)))
for invalid in (list(range(127)), list(range(129)), [True]*128, [-1]*128, [151936]*128):
    rejected(lambda: runtime.validate_inputs(invalid), 'Token' if len(invalid)==128 else '128')
rejected(lambda: runtime.validate_inputs(list(range(128)),31), '32')
checks.append('fixed_scope_input_guards')

events = []
torch = ModuleType('torch')
torch.float16 = 'float16'
torch.__version__ = '2.9.1+cu128'
torch.version = NS(cuda='12.8')


def device_name(index):
    assert events and events[0]=='engine'
    events.append('gpu_query')
    return 'NVIDIA GeForce RTX 4070 Ti'


torch.cuda = NS(get_device_name=device_name, synchronize=lambda: events.append('sync'))
minisgl = ModuleType('minisgl')
minisgl.__path__ = []
llm = ModuleType('minisgl.llm')
core = ModuleType('minisgl.core')
core.SamplingParams = NS
kernel = ModuleType('minisgl.kernel')
store = ModuleType('minisgl.kernel.store')
store.store_cache = kernel.store_cache = object()
minisgl.kernel = kernel


class LLM:
    def __init__(self, path, dtype, **options):
        events.append('engine')
        self.engine = NS(graph_runner=NS(graph_bs_list=options['cuda_graph_bs']))
    def generate(self, prompts, params):
        self.offline_send_result([NS(next_token=token) for token in range(32)])
        return [{'token_ids': list(range(32)), 'text':'fake'}]
    def offline_send_result(self, reply):
        pass
    def shutdown(self):
        events.append('shutdown')


llm.LLM = LLM
runtime.verify_environment = lambda project, model_path: {'project':'/fake/project','model_path':'/fake/model',
    'packages':runtime.CONTRACT['timing_packages']}
# Actual Python import consults real sys.modules; the separate mock below exposes
# an empty pre-import module inventory to the source's fresh-process assertion.
runtime.sys = NS(modules={}, path=[])
with patch.dict(sys.modules, {'torch':torch,'minisgl':minisgl,'minisgl.llm':llm,'minisgl.core':core,
                             'minisgl.kernel':kernel,'minisgl.kernel.store':store}):
    for execution in ('eager','decode_graph'):
        events.clear()
        with runtime.StableRuntime(execution=execution) as instance:
            assert instance.model.engine.graph_runner.graph_bs_list == ([1] if execution=='decode_graph' else [])
            assert instance.generate()[0]['token_ids'] == list(range(32))
            assert instance.raw_token_ids == list(range(32))
            instance.generate()
            assert instance.raw_token_ids == list(range(32)), 'Raw outputs must reset per request'
            instance._lock.acquire()
            rejected(instance.generate, 'Concurrent')
            instance._lock.release()
            instance.synchronize()
        assert events == ['engine','gpu_query','sync','shutdown']
    checks.append('persistent_generate_reset_exclusive_access_explicit_execution_and_post_engine_device_query')
    def broken_engine(*args, **kwargs):
        events.append('failed_engine')
        raise RuntimeError('Deliberate engine failure')
    llm.LLM = broken_engine
    # A callable cannot be subclassed, so use an actual failing class.
    class FailedLLM(LLM):
        def __init__(self,*args,**kwargs):
            events.append('failed_engine')
            raise RuntimeError('Deliberate engine failure')
    llm.LLM = FailedLLM
    events.clear()
    rejected(lambda: runtime.StableRuntime(), 'Deliberate engine failure')
    assert events == ['failed_engine'], 'No retry/fallback engine may be instantiated'
    checks.append('engine_failure_does_not_retry_or_fallback')

with tempfile.TemporaryDirectory(prefix='stable54-cpu-') as temp:
    run = Path(temp)
    for policy in ('eager','decode_graph'):
        (run/policy).mkdir()
        runner.save(run/policy/'result.json', {'status':'STABLE_RUNTIME_LOGITS_COLLECTED',
                    'raw_generated_token_ids':list(range(32))})
    values = {f'step{i:02d}':np.ones((1,151936),dtype=np.float32) for i in range(32)}
    for policy in ('eager','decode_graph'):
        np.savez(run/policy/'all-logits.npz',**values)
    result = runner.compare_logits(run,list(range(32)))
    assert result['pass'] and result['elements_compared']==4861952
    values['step31'][0,151935] += 0.01
    np.savez(run/'decode_graph/all-logits.npz',**values)
    result = runner.compare_logits(run,list(range(32)))
    assert not result['pass'] and result['violations']==1
    values['step31'][0,151935] = np.nan
    np.savez(run/'decode_graph/all-logits.npz',**values)
    assert not runner.compare_logits(run,list(range(32)))['pass']
    checks.append('all32logits_comparator_pass_single_violation_and_nonfinite_controls')

result = {'status':'STABLE_RUNTIME_CPU_CONTROLS_PASS','checks':checks,
          'runtime_sha256':runner.sha(base/'53-m08-stable-runtime.py'),
          'runner_sha256':runner.sha(base/'54-verify-stable-runtime.py'),
          'limit':'CPU synthetic/fake-object controls only; no real Engine, CUDA run, speed or numerical model acceptance.'}
runner.save(Path(__file__).with_suffix('.json'), result)
print(json.dumps(result,indent=2))
