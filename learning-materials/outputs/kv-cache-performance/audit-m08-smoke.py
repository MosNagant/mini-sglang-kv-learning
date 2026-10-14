"""CPU evidence audit for an actually completed 39 run, no model execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('run',type=Path)
a=ap.parse_args()
run=a.run
result=json.loads((run/'result.json').read_text(encoding='utf-8'))
request=json.loads((run/'request.json').read_text(encoding='utf-8'))
manifest=json.loads((run/'source-manifest.json').read_text(encoding='utf-8'))
assert result['status']=='MODEL_SMOKE_AND_KV_TRACE_COMPLETE'
for name,digest in manifest.items():
    assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((run/'runner.py').read_bytes()).hexdigest()==request['runner_sha256']
trace=[json.loads(line) for line in (run/'kv-trace.jsonl').read_text(encoding='utf-8').splitlines()]
done=[r for r in trace if r['status']=='ORIGINAL_CALL_RETURNED']
assert len(trace)==2*len(done)==448
assert len({r['call_id'] for r in done})==224
assert Counter((r['phase'],r['layer']) for r in done)==Counter({(phase,layer):n for phase,n in [('prefill',1),('decode',7)] for layer in range(28)})
for r in done:
    assert r['flattened_D']==1024
    b=5 if r['phase']=='prefill' else 1
    assert r['tokens_B']==b
    assert len(r['indices'])==len(set(r['indices']))==b
    assert all(0<=i<257 for i in r['indices'])
    for name in ['k','v']:
        t=r['tensors'][name]
        assert t['shape']==[b,1024] and t['stride']==[4096,1] and t['dtype']=='torch.float16'
    for name in ['k_cache','v_cache']:
        t=r['tensors'][name]
        assert t['shape']==[257,8,128] and t['stride']==[1024,128,1] and t['dtype']=='torch.float16'
    assert r['tensors']['indices']['dtype']=='torch.int32'
tokens=json.loads((run/'raw-output-tokens.json').read_text(encoding='utf-8'))
assert [t['token_id'] for t in tokens]==result['raw_generated_token_ids']
assert result['outputs'][0]['token_ids']==result['raw_generated_token_ids']
audit={'status':'SMOKE_EVIDENCE_AUDIT_PASS','source_files_verified':len(manifest),'completed_original_calls':len(done),'raw_tokens':len(tokens),'scope':'Artifact integrity and recorded call metadata; no independent logits/reference correctness or timing claim.'}
(run/'evidence-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit))
