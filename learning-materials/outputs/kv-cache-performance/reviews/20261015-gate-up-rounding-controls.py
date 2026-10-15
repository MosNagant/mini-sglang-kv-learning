"""Independent small CPU controls of integer-dot and FP16 ties-to-even audit."""
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('20261015-gate-up-rounding-analysis.py')
spec=importlib.util.spec_from_file_location('rounding_audit_cpu_control',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cases=[([1,1],[1,2**-11],1),([1,1],[1,3*2**-11],1+2**-9),
       ([1,1],[-1,-2**-11],-1),([1,1],[-1,-3*2**-11],-1-2**-9),
       ([2**-24],[.5],0),([-2**-24],[.5],-0.0),
       ([65504,65504],[1,-1],0),([.25,.5,1],[-.125,.25,-.5],-.40625)]
records=[]
for values,weights,expected in cases:
    x,w=np.array(values,dtype=np.float16),np.array(weights,dtype=np.float16)
    exact=module.exact_dot(x,w)
    reference=sum((Fraction(float(a))*Fraction(float(b)) for a,b in zip(x,w)),Fraction())
    assert Fraction(exact,2**48)==reference
    rounded,metadata=module.correct_round(exact)
    assert module.bits(rounded)==module.bits(np.float16(expected)), (values,weights,rounded,expected)
    records.append({'x':values,'w':weights,'expected_fp16_bits':module.bits(np.float16(expected)),
                    'actual_fp16_bits':module.bits(rounded),'exact_dot_matches_fraction':True,
                    'midpoint':metadata['exact_midpoint']})
assert records[0]['midpoint'] and records[1]['midpoint'] and records[4]['midpoint']
result={'status':'EXACT_ROUNDING_CPU_CONTROLS_PASS','gpu_execution':False,'controls':records,
        'analysis_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'limit':'Finite-output audit cases including positive/negative midpoints, signed zero and cancellation; not GPU implementation tests.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'controls':len(records)}))
