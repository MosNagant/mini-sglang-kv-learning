"""Independent CPU review of the actual lesson kernel body; no GPU execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RUNNER = BASE / "21-check-kv-mask.py"
SOURCE = BASE / "21-triton-store-mask.py"
watched = [BASE / n for n in ("14-check-kv-widths.py", "14-check-kv-widths.sh", "21-check-kv-mask.py", "21-check-kv-mask.sh", "21-triton-store-mask.py")]
def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
start_hashes = hashes()
source = SOURCE.read_text(encoding="utf-8")
spec = importlib.util.spec_from_file_location("independent_mask_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class Ptr:
    def __init__(self, data, origin, tag, stride):
        self.data, self.origin, self.tag, self.stride = data, origin, tag, stride
    def __add__(self, offsets):
        return Addr(self, offsets)

class Addr:
    def __init__(self, ptr, offsets):
        self.ptr = ptr
        self.offsets = np.asarray(offsets, dtype=np.int64)

class TL:
    constexpr = int
    def __init__(self, width):
        self.pid, self.width, self.ops = 0, width, []
    def program_id(self, axis):
        assert axis == 0
        return self.pid
    def arange(self, lo, hi):
        return np.arange(lo, hi, dtype=np.int64)
    def active(self, addr, mask, kind, other=None):
        selected = np.ones(addr.offsets.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        assert selected.shape == addr.offsets.shape
        absolute = addr.ptr.origin + addr.offsets
        active = absolute[selected]
        assert np.all(active >= 0) and np.all(active < addr.ptr.data.size), "CPU_EMULATED_ALLOCATION_BOUNDS"
        logical_invalid = 0
        if addr.ptr.tag != "indices":
            columns = addr.offsets[selected] % addr.ptr.stride
            logical_invalid = int(np.count_nonzero(columns >= self.width))
        self.ops.append(dict(kind=kind, tag=addr.ptr.tag, pid=self.pid,
                             mask_present=mask is not None, active=int(selected.sum()),
                             inactive=int(selected.size-selected.sum()),
                             logical_invalid=logical_invalid, other=other,
                             col191=bool(selected[191]) if selected.ndim and selected.size > 191 else None,
                             col192=bool(selected[192]) if selected.ndim and selected.size > 192 else None))
        return selected, active
    def load(self, addr, mask=None, other=None):
        selected, active = self.active(addr, mask, "load", other)
        out = np.full(addr.offsets.shape, 0 if other is None else other, dtype=addr.ptr.data.dtype)
        out[selected] = addr.ptr.data[active]
        return out.item() if out.ndim == 0 else out
    def store(self, addr, values, mask=None):
        selected, active = self.active(addr, mask, "store")
        values = np.asarray(values)
        assert values.shape == selected.shape
        addr.ptr.data[active] = values[selected]

def simulate(cfg, kernel_source):
    d, gap, guard, layout = (cfg[x] for x in ("width", "input_gap", "cache_guard", "layout"))
    pitch = d + guard
    shape = (8,2,pitch) if layout == "interleaved" else (2,8,pitch)
    cache = np.full(shape, -1, dtype=np.float16)
    if layout == "interleaved":
        cache[:,:,d:] = -7
    else:
        cache[:,:,d:] = -7
    k = np.full((3,d+gap), -9, dtype=np.float16)
    v = np.full((3,d+gap), -9, dtype=np.float16)
    for p in range(3):
        for c in range(d):
            k[p,c] = 10 + p*d + c
            v[p,c] = 1010 + p*d + c
    rows = np.array([7,0,3], dtype=np.int32)
    before = dict(cache=cache.copy(), k=k.copy(), v=v.copy(), indices=rows.copy())
    expected = cache.copy()
    for p,pos in enumerate(rows):
        if layout == "interleaved":
            expected[pos,0,:d],expected[pos,1,:d] = k[p,:d],v[p,:d]
        else:
            expected[0,pos,:d],expected[1,pos,:d] = k[p,:d],v[p,:d]
    before['expected_cache'] = expected
    fixture = runner.fixture(cfg)
    for key,val in before.items():
        np.testing.assert_array_equal(val,fixture[key],err_msg="INDEPENDENT_FIXTURE_"+key)
    src = d + gap
    dst = pitch * (2 if layout == "interleaved" else 1)
    vorigin = pitch if layout == "interleaved" else 8*pitch
    tl = TL(d)
    fn = next(n for n in ast.parse(kernel_source).body if isinstance(n,ast.FunctionDef) and n.name == "store_kv_v1")
    ns = dict(tl=tl,triton=SimpleNamespace(jit=lambda f:f))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),ns)
    ptrs = [Ptr(cache.reshape(-1),0,'k_cache',dst),Ptr(cache.reshape(-1),vorigin,'v_cache',dst),
            Ptr(rows.reshape(-1),0,'indices',1),Ptr(k.reshape(-1),0,'k_input',src),Ptr(v.reshape(-1),0,'v_input',src)]
    for p in range(3):
        tl.pid = p
        ns['store_kv_v1'](*ptrs,WIDTH=d,SRC_ROW_STRIDE=src,DST_ROW_STRIDE=dst,BLOCK_SIZE=1<<(d-1).bit_length())
    after = dict(cache=cache,k=k,v=v,indices=rows)
    correct = True
    try:
        runner.h.verify_backing(before,after)
    except AssertionError:
        correct = False
    return correct,tl.ops

configs = runner.configurations()
assert len(configs) == 14
assert len({tuple(sorted(c.items())) for c in configs}) == 14
positive = []
for cfg in configs:
    correct,ops = simulate(cfg,source)
    assert correct
    vector = [x for x in ops if x['tag'] != 'indices']
    assert len(vector) == 12 and all(x['mask_present'] and x['logical_invalid']==0 for x in vector)
    assert all(x['other']==0 for x in vector if x['kind']=='load')
    assert all(x['active']==cfg['width'] for x in vector)
    if cfg['width'] == 192:
        assert all(x['col191'] is True and x['col192'] is False and x['inactive']==64 for x in vector)
    positive.append(dict(**cfg,status='CPU_KERNEL_BODY_PASS',logical_active=cfg['width'],logical_block=1<<(cfg['width']-1).bit_length()))
review = runner.validate_mask_source(source)
mutations = {
 'missing_K_load_mask':source.replace('tl.load(k_ptr + src_offsets, mask=mask, other=0)','tl.load(k_ptr + src_offsets, other=0)'),
 'missing_V_load_mask':source.replace('tl.load(v_ptr + src_offsets, mask=mask, other=0)','tl.load(v_ptr + src_offsets, other=0)'),
 'missing_K_store_mask':source.replace('tl.store(k_cache_ptr + dst_offsets, k_values, mask=mask)','tl.store(k_cache_ptr + dst_offsets, k_values)'),
 'missing_V_store_mask':source.replace('tl.store(v_cache_ptr + dst_offsets, v_values, mask=mask)','tl.store(v_cache_ptr + dst_offsets, v_values)'),
 'mask_includes_width':source.replace('mask = cols < WIDTH','mask = cols <= WIDTH'),
 'wrong_source_stride':source.replace('pid * SRC_ROW_STRIDE','pid * WIDTH'),
 'wrong_destination_stride':source.replace('pos * DST_ROW_STRIDE','pos * WIDTH'),
}
negative=[]
guard_cfg=dict(width=192,layout='interleaved',input_gap=64,cache_guard=64)
for name,bad in mutations.items():
    assert bad != source
    try: runner.validate_mask_source(bad)
    except RuntimeError: pass
    else: raise AssertionError('PREFLIGHT_ACCEPTED_'+name)
    correct,ops=simulate(guard_cfg,bad)
    invalid_reads=sum(x['logical_invalid'] for x in ops if x['kind']=='load')
    invalid_writes=sum(x['logical_invalid'] for x in ops if x['kind']=='store')
    if name in ('missing_K_load_mask','missing_V_load_mask'):
        assert correct and invalid_reads == 192 and invalid_writes == 0
    else:
        assert not correct
    negative.append(dict(mutation=name,preflight_rejected=True,full_storage_would_pass=correct,
                         cpu_emulated_invalid_logical_reads=invalid_reads,cpu_emulated_invalid_logical_writes=invalid_writes))
controls=runner.storage_controls()
assert len(controls)==14 and sum(len(x['negative_controls']) for x in controls)==122
assert hashes()==start_hashes
result=dict(status='CPU_INDEPENDENT_REVIEW_PASS',GPU='NOT_RUN',Triton_compilation='NOT_RUN',
            method='Execute actual Python kernel function body with NumPy-backed pointer/mask emulation; independently construct fixtures; static negative controls',
            source_hashes=start_hashes,source_review=review,positive_kernel_configurations=positive,
            negative_kernel_controls=negative,storage_oracle=dict(positive=14,negative=122),
            limitations=['CPU emulation is not GPU execution, compilation, a sanitizer or a benchmark.',
                         'Unmasked allocated guard reads can leave all arrays unchanged; static mask review is required.',
                         'Does not claim LLVM/PTX code generation or runtime invalid-read detection.'])
out=Path(__file__).with_suffix('.json')
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],GPU='NOT_RUN',kernel_fixtures=14,negative_kernel_controls=7,
                     storage_positive=14,storage_negative=122,source_unchanged=True,report=str(out))))
