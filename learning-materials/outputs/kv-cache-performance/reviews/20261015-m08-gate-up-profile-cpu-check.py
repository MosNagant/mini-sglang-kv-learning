"""CPU controls for 52's metric parser and error classification; no GPU calls."""
import ast
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
source=root/'52-m08-gate-up-profile.py'
ast.parse(source.read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('profile52_cpu',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
buf=io.StringIO();writer=csv.writer(buf)
writer.writerow(['ID','Kernel Name','Metric Name','Metric Unit','Metric Value'])
for identity in ('0','1'):
    for metric in module.METRICS:
        writer.writerow([identity,'actual_kernel',metric,'unit','1,024'])
text=buf.getvalue();parsed=module.parse_raw(text)
assert parsed['kernel_count']==2 and len(parsed['records'])==8
assert all(x['value']==1024 for x in parsed['records'])
negative_controls=[]
for name,bad in [('missing_header','no CSV'),('missing_metric',text.replace(module.METRICS[0],'other')),
                 ('nan',text.replace('1,024','nan')),('negative',text.replace('1,024','-1'))]:
    try:module.parse_raw(bad)
    except (ValueError,RuntimeError):negative_controls.append(name)
    else:raise AssertionError('Accepted '+name)
assert module.classify('==ERROR== ERR_NVGPUCTRPERM - permission denied',1)=='GPU_COUNTER_PERMISSION_DENIED'
assert module.classify('Profiling failed because a driver resource was unavailable. https://developer.nvidia.com/ERR_NVGPUCTRPERM',1)=='DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE'
assert module.classify('Help: https://developer.nvidia.com/ERR_NVGPUCTRPERM',1)=='FAILED_EXIT_1'
assert module.classify('ok',0)=='COMMAND_PASS'
record={'status':'CPU_CONTROLS_PASS','gpu_execution':False,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'shell_sha256':hashlib.sha256(source.with_suffix('.sh').read_bytes()).hexdigest(),
        'positive_metric_rows':8,'negative_controls':negative_controls,
        'classification_controls':4,
        'limit':'Parser/control-flow only. Does not verify ncu installation, supported metric names, CUDA execution, counters or replay behavior.'}
destination=Path(__file__).with_suffix('.json');destination.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
