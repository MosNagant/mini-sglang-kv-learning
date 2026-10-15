"""CPU-only source/AST/ticket controls. No torch/minisgl import or GPU work."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE/'stable/graph_greedy.py'
SOURCE = BASE/'logs/m08-gate-up-model-check-results/run-20261015-044602-wwyl5b3t/source/python/minisgl'


class Tensor:
    def __init__(self, *, shape=(1,151936), stride=(151936,1), dtype='torch.float32',
                 device='cuda:0', pointer=1024, offset=0):
        self.shape, self._stride, self.dtype = shape, stride, dtype
        self.device, self.pointer, self.offset = device, pointer, offset

    def stride(self): return self._stride
    def data_ptr(self): return self.pointer
    def storage_offset(self): return self.offset


def main():
    spec=importlib.util.spec_from_file_location('graph_greedy_cpu_control', MODULE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results=[]
    def check(name, condition):
        if not condition: raise AssertionError(name)
        results.append({'name':name,'pass':True})
    for name, digest in module.SOURCE_SHA256.items():
        check('source_sha256_'+name, hashlib.sha256((SOURCE/'engine'/name).read_bytes()).hexdigest()==digest)
    source=(SOURCE/'engine/graph.py').read_text(encoding='utf-8')
    transformed=module._transformed_capture(source)
    compile(transformed,'cpu-transformed-capture','exec')
    nodes=[n for n in ast.walk(transformed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='_graph_greedy_capture']
    check('exact_two_capture_insertions',len(nodes)==2)
    # Remove only the inserted statements: original capture AST must be identical.
    class Remove(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_graph_greedy_capture': return None
            return node
    restored=Remove().visit(transformed).body[1]
    original=next(n for n in next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='GraphRunner').body
                  if isinstance(n,ast.FunctionDef) and n.name=='_capture_graphs')
    check('all_original_capture_operations_unchanged',ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False))
    try:
        module._transformed_capture(source.replace('self.buffer.logits[:bs] = model.forward()','self.buffer.logits[:bs] = changed.forward()',1))
    except RuntimeError: check('changed_capture_structure_rejected',True)
    else: check('changed_capture_structure_rejected',False)
    check('ENV_import_exists', 'from minisgl.env import ENV' in MODULE.read_text(encoding='utf-8'))
    info=ast.parse((SOURCE/'distributed/info.py').read_text(encoding='utf-8'))
    check('TP_size_exists',any(isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='size' for n in ast.walk(info)))
    sampler=SimpleNamespace(vocab_size=151936,device='cuda:0')
    greedy=SimpleNamespace(temperatures=None,top_k=object(),top_p=object())
    logits=Tensor()
    tokens=Tensor(shape=(1,),stride=(1,),dtype='torch.int32',pointer=2048)
    ticket={'signature':module._signature(logits),'tokens':tokens,'token_signature':module._signature(tokens),'stream':'engine-stream','generation':1}
    def reason(value=logits,args=greedy,sample=sampler,stream='engine-stream',current=ticket):
        return module._reuse_reason(current,value,args,sample,stream)
    check('matching_new_view_accepted',reason(Tensor()) is None)
    check('top_k_top_p_ignored_for_greedy_like_original',reason() is None)
    for name,value in [('pointer',Tensor(pointer=8192)),('shape',Tensor(shape=(2,151936))),
                       ('stride',Tensor(stride=(151937,1))),('dtype',Tensor(dtype='torch.float16')),
                       ('device',Tensor(device='cuda:1')),('storage_offset',Tensor(offset=1))]:
        check('changed_'+name+'_rejected',reason(value)=='logits_metadata_mismatch')
    check('non_greedy_rejected',reason(args=SimpleNamespace(temperatures=object()))=='non_greedy')
    check('missing_ticket_rejected',reason(current=None)=='no_replay_ticket')
    check('wrong_stream_rejected',reason(stream='other-stream')=='stream_mismatch')
    check('sampler_device_rejected',reason(sample=SimpleNamespace(device='cuda:1',vocab_size=151936))=='sampler_device_mismatch')
    check('sampler_vocab_rejected',reason(sample=SimpleNamespace(device='cuda:0',vocab_size=32))=='sampler_shape_mismatch')
    changed=dict(ticket,tokens=Tensor(shape=(1,),stride=(1,),dtype='torch.int32',pointer=4096))
    check('changed_token_storage_rejected',reason(current=changed)=='captured_token_storage_changed')
    handle=module.PatchHandle.__new__(module.PatchHandle)
    handle.counts={'sample_hits':0,'original_samples':0}
    handle.fallback_reasons={}
    handle.pending=ticket
    match,value=handle._take_token(sampler,Tensor(),greedy,'engine-stream')
    check('one_ticket_hit_returns_same_persistent_tokens',match and value is tokens and handle.pending is None)
    match,why=handle._take_token(sampler,Tensor(),greedy,'engine-stream')
    check('second_sample_cannot_reuse_stale_tokens',not match and why=='no_replay_ticket')
    handle.pending=ticket
    match,why=handle._take_token(sampler,Tensor(pointer=8192),greedy,'engine-stream')
    check('mismatch_also_consumes_ticket',not match and handle.pending is None)
    check('counts_record_all_paths',handle.counts=={'sample_hits':1,'original_samples':2})
    result={'status':'CPU_CONTROLS_PASS','controls':results,'count':len(results),
            'module_sha256':hashlib.sha256(MODULE.read_bytes()).hexdigest(),
            'source_sha256':module.SOURCE_SHA256,
            'limit':'Metadata/AST/source checks only. No CUDA execution, captured-operation equivalence or speed validation.'}
    out=Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'count':len(results),'saved':str(out)},ensure_ascii=False))


if __name__=='__main__': main()
