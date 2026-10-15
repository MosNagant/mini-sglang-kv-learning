"""CPU-only controls for55's protocol/comparator; no torch/minisgl or GPU."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
from unittest.mock import patch
import numpy as np

BASE=Path(__file__).resolve().parents[1]


def main():
    path=BASE/'55-verify-graph-greedy.py'
    spec=importlib.util.spec_from_file_location('graph_greedy_protocol_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checks=[]
    def check(name,condition):
        if not condition: raise AssertionError(name)
        checks.append({'name':name,'pass':True})
    source=(BASE/'53-m08-stable-runtime.py').read_text(encoding='utf-8')
    transformed=module.runtime_with_hook(source)
    check('one_controlled_hook_only',transformed.replace('        _EXPERIMENT_INSTALL_HOOK()  # 55: after source guards, before Engine\n','')==source)
    compile(transformed,'control-runtime','exec')
    check('runtime_ast_compiles',True)
    try:module.runtime_with_hook(source.replace(module.INSERTION_ANCHOR,''))
    except RuntimeError:check('missing_hook_anchor_rejected',True)
    else:check('missing_hook_anchor_rejected',False)
    tokens=[1]*32
    cfg={'contract':{'raw_expected_token_ids':tokens},'strict_requests':[
        {'name':'A','input_token_ids':[1]*128},{'name':'B','input_token_ids':[2]*128},{'name':'A','input_token_ids':[1]*128}]}
    workers={};arrays={}
    class Archive:
        def __init__(self,data):self.data=data;self.files=list(data)
        def __getitem__(self,key):return self.data[key]
        def __enter__(self):return self
        def __exit__(self,*args):pass
    zero=np.zeros((1,151936),dtype=np.float32)
    for arm in module.ARMS:
        requests=[]
        for index,request in enumerate(cfg['strict_requests']):
            name=f'r{index}.npz'
            requests.append(dict(request,logits_file=name,logits_sha256='mock-hash',raw_generated_token_ids=list(tokens)))
            arrays[('run',arm,name)]={f'step{s:02d}':zero for s in range(32)}
        workers[('run',arm,'result.json')]={'status':'GRAPH_GREEDY_LOGITS_COLLECTED','requests':requests}
    def load_mock(path,**kwargs):return Archive(arrays[Path(path).parts])
    def compare():
        with patch.object(module,'read',side_effect=lambda p:workers[Path(p).parts]),patch.object(module,'sha',return_value='mock-hash'),patch.object(np,'load',side_effect=load_mock):
            return module.compare_logits(Path('run'),cfg)
    baseline=compare()
    check('all96_matching_steps_accepted',baseline['pass'] and baseline['elements_compared']==96*151936)
    key=('run','graph_greedy','r1.npz')
    changed=zero.copy();changed[0,0]=-0.0
    arrays[key]['step00']=changed
    result=compare()
    check('one_signed_zero_bit_difference_rejected',not result['pass'] and result['bit_differences']==1 and result['steps'][32]['max_abs_error']==0.)
    arrays[key]['step00']=zero
    workers[('run','graph_greedy','result.json')]['requests'][1]['raw_generated_token_ids']=[2]*32
    check('B_tokens_different_rejected',not compare()['pass'])
    workers[('run','graph_greedy','result.json')]['requests'][1]['raw_generated_token_ids']=list(tokens)
    changed=zero.copy();changed[0,0]=float('nan')
    arrays[key]['step00']=changed
    check('nonfinite_logits_rejected',not compare()['pass'])
    arrays[key]['step00']=zero
    # Both arms agree with each other on a changed final A, but A->B->A fails.
    changed=zero.copy();changed[0,0]=1.
    for arm in module.ARMS:arrays[('run',arm,'r2.npz')]['step00']=changed
    result=compare()
    check('same_cross_arm_but_changed_repeated_A_rejected',not result['pass'] and result['bit_differences']==0)
    result={'status':'CPU_PROTOCOL_CONTROLS_PASS','count':len(checks),'controls':checks,
            'runner_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'limit':'Synthetic in-memory arrays/mock file hashes test comparator logic only; actual evidence hashes/Graph execution require real run.'}
    out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'count':len(checks),'saved':str(out)}))


if __name__=='__main__':main()
