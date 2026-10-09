import importlib.util, json, pathlib, sys, tempfile
sys.dont_write_bytecode=True
base=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('wrapper_runner_local',base/'22-check-kv-wrapper.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
folder=pathlib.Path(tempfile.mkdtemp(prefix='m06-wrapper-install-',dir=base/'reviews')).resolve()
assert folder.parent == (base/'reviews').resolve()
exp=folder/'experiments';exp.mkdir()
(exp/'triton_store_mask.py').write_text('accepted mask sentinel\n',encoding='utf-8')
(exp/'triton_store_v1.py').write_text('old v1 sentinel\n',encoding='utf-8')
r.prepare(folder)
assert (exp/'triton_store_api.py').read_text(encoding='utf-8') == (base/'22-triton-store-api-template.py').read_text(encoding='utf-8')
(exp/'triton_store_api.py').write_text('learner sentinel\n',encoding='utf-8')
(exp/'kv_contract_support.py').write_text('existing support sentinel\n',encoding='utf-8')
r.prepare(folder)
assert (exp/'triton_store_api.py').read_text(encoding='utf-8') == 'learner sentinel\n'
assert (exp/'kv_contract_support.py').read_text(encoding='utf-8') == 'existing support sentinel\n'
assert (exp/'triton_store_mask.py').read_text(encoding='utf-8') == 'accepted mask sentinel\n'
assert (exp/'triton_store_v1.py').read_text(encoding='utf-8') == 'old v1 sentinel\n'
report=dict(status='INSTALL_PRESERVATION_PASS', GPU='NOT_RUN', evidence=str(folder), checks=['new template installed with blanks','existing learner kept','existing support kept','mask unchanged','old v1 unchanged'])
(base/'reviews/20261009-m06-04-install-review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
