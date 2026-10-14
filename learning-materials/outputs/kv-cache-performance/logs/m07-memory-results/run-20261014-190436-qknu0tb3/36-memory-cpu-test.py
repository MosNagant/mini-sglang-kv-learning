"""CPU-only controls for36: actual view builder, calls, rotation, oracle and gates.

Usage: python -B 36-memory-cpu-test.py NEW_REPORT_PATH
Does not import actual torch/triton, call WSL, execute GPU or overwrite reports.
"""
import ast
import copy
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("memory36_under_test",HERE/"36-compare-kv-memory.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class Device:
    type = "cuda"
    def __str__(self): return "cuda:0"


class Tensor:
    next_ptr = 0x10000000
    def __init__(self,array,root=None,base=None):
        self.array = array
        self.root = array if root is None else root
        if base is None:
            base = Tensor.next_ptr
            Tensor.next_ptr += ((array.nbytes+255)//256+1)*256
        self.base = base
        self.device = DEVICE
        self.dtype = "torch.int32" if array.dtype == np.int32 else "torch.float16"
    @property
    def shape(self): return self.array.shape
    @property
    def ndim(self): return self.array.ndim
    def stride(self,index=None):
        value = tuple(x//self.array.itemsize for x in self.array.strides)
        return value if index is None else value[index]
    def data_ptr(self): return self.base+self.array.ctypes.data-self.root.ctypes.data
    def storage_offset(self): return (self.data_ptr()-self.base)//self.array.itemsize
    def numel(self): return self.array.size
    def element_size(self): return self.array.itemsize
    def cuda(self): return self
    def cpu(self): return self
    def tolist(self): return self.array.tolist()
    def as_strided(self,shape,stride,storage_offset):
        view = np.lib.stride_tricks.as_strided(self.root[storage_offset:],shape=shape,
            strides=tuple(x*self.array.itemsize for x in stride),writeable=True)
        return Tensor(view,self.root,self.base)
    def __getitem__(self,key): return Tensor(self.array[key],self.root,self.base)


DEVICE = Device()
class Torch:
    Tensor = Tensor
    float16,int32 = "torch.float16","torch.int32"
    @staticmethod
    def from_numpy(array): return Tensor(array)


def cpu_copy(kc,vc,indices,k,v):
    kc.array[indices.array,:] = k.array
    vc.array[indices.array,:] = v.array


class Kernel:
    def __init__(self,arm,cfg): self.arm,self.cfg,self.seen = arm,cfg,[]
    def __getitem__(self,launch_grid):
        def call(*args,**kwargs):
            assert launch_grid == t.grid(self.arm) and kwargs == t.parameters(self.cfg,self.arm)
            self.seen.append(tuple(x.data_ptr() for x in args))
            cpu_copy(*args)
            return self.arm
        return call


def geometry(cfg):
    packets,records = [],[]
    for slot in range(cfg["packets"]):
        before = t.fixture(cfg,slot)
        buffers,args = t.gpu_packet(Torch,cfg,before)
        packets.append((before,buffers,args))
        records.append(dict(slot=slot,buffers={name:t.tensor_record(x) for name,x in buffers.items()},
                            views=[t.tensor_record(x) for x in args]))
    return packets,records


def main(output):
    if output.exists(): raise FileExistsError("REPORT_EXISTS")
    rejected = []
    def reject(label,call,error=ValueError):
        try: call()
        except error: rejected.append(label)
        else: raise AssertionError("NEGATIVE_CONTROL_MISSED: "+label)
    plan = t.make_plan()
    assert len(plan["schedule"]) == 18 and plan["interval_averages"] == 3240
    for round_id in (1,2,3):
        positions = Counter((item,index) for sample in range(30)
            for index,item in enumerate(t.measurement_order(sample,round_id)))
        assert len(positions) == 36 and set(positions.values()) == {5}
    records_by_condition = {}
    fixtures = []
    for condition in t.CASES:
        cfg = t.configuration(condition)
        packets,records = geometry(cfg)
        records_by_condition[condition] = records
        t.validate_address_records(cfg,records)
        assert sorted(cfg["rows"]) == list(range(t.B))
        for slot,(before,buffers,args) in enumerate(packets):
            assert args[3].storage_offset() == (68 if condition == "E" else 64)
            assert args[3].stride() == (192 if condition == "B" else 128,1)
            after = {name:before["expected_cache" if name == "cache" else name].copy() for name in t.BUFS}
            t.verify_arrays(cfg,slot,before,after)
            targets = [("cache",(0,0,0)),("k",0),("k",len(before["k"])-1),("v",0),("indices",0)]
            if condition == "B": targets.append(("k",64+t.D))
            for name,index in targets:
                bad = {key:value.copy() for key,value in after.items()}
                bad[name][index] = -99
                reject(f"{condition}/{slot}/{name}/{index}",lambda:t.verify_arrays(cfg,slot,before,bad))
            bad_before = {key:value.copy() for key,value in before.items()}
            bad_before["indices"][0] = bad_before["indices"][1]
            reject(f"{condition}/{slot}/duplicate-index",lambda:t.verify_arrays(cfg,slot,bad_before,after))
        for arm in t.ARMS:
            kernels = [Kernel(arm,cfg) for _ in packets]
            calls = [t.raw_call(cpu_copy,kernel,cfg,arm,args) for kernel,(_,_,args) in zip(kernels,packets)]
            cycle = t.CaptureCycle(calls,cfg)
            for _ in range(100): cycle()
            assert cycle.trace == t.slot_table(cfg)*25 and cycle.position == 0
            for slot,(before,buffers,args) in enumerate(packets):
                t.verify_arrays(cfg,slot,before,{name:x.array.copy() for name,x in buffers.items()})
                if arm != "original_cuda":
                    expected_calls = 25 if condition == "F" else 100
                    assert len(kernels[slot].seen) == expected_calls
                    assert set(kernels[slot].seen) == {tuple(x.data_ptr() for x in args)}
                buffers["cache"].array[...] = before["cache"]
        bad = copy.deepcopy(records); bad[0]["views"][3]["storage_offset"] = 0
        reject(condition+"/lost-offset",lambda:t.validate_address_records(cfg,bad))
        bad = copy.deepcopy(records); bad[0]["views"][3]["stride"][0] += 1
        reject(condition+"/bad-stride",lambda:t.validate_address_records(cfg,bad))
        bad = copy.deepcopy(records); bad[0]["views"][3]["data_ptr"] += 4
        reject(condition+"/misaligned-view",lambda:t.validate_address_records(cfg,bad))
        fixtures.append(dict(condition=condition,packets=cfg["packets"],source_offset=cfg["source_offset_elements"],
                             input_mod128=cfg["input_address_mod128"],full_backing_and_real_factory_mock="PASS"))
    cfg = t.configuration("F")
    bad = copy.deepcopy(records_by_condition["F"]); bad[1] = copy.deepcopy(bad[0]); bad[1]["slot"] = 1
    reject("F/alias-packet",lambda:t.validate_address_records(cfg,bad))
    require_A = t.fixture(t.configuration("A"),0)
    for condition in "BCDE":
        cfg = t.configuration(condition); before = t.fixture(cfg,0)
        a = np.stack([require_A["k"][64+i*128:64+i*128+128] for i in range(t.B)])
        b = np.stack([before["k"][cfg["source_offset_elements"]+i*(128+cfg["gap"]):cfg["source_offset_elements"]+i*(128+cfg["gap"])+128] for i in range(t.B)])
        np.testing.assert_array_equal(a,b)
    assert not np.array_equal(t.fixture(t.configuration("F"),0)["k"],t.fixture(t.configuration("F"),1)["k"])
    synthetic = []
    for item in plan["schedule"]:
        condition,round_id = item["condition"],item["round"]
        cfg = t.configuration(condition)
        factor = 1.1 if condition == "F" else 1
        samples = {arm:{policy:[.001*factor*(1 if arm == "original_cuda" else .8)*(2 if policy == "eager" else 1)*(1+(n-14.5)*.001) for n in range(30)] for policy in t.POLICIES} for arm in t.ARMS}
        measurements = [dict(sample=n,implementation=arm,policy=policy,ms_per_call=samples[arm][policy][n])
            for n in range(30) for arm,policy in t.measurement_order(n,round_id)]
        synthetic.append(dict(item,status="COLLECTED",config=cfg,calls_per_interval=100,
            correctness_checks=190,packet_checks=190*cfg["packets"],buffer_checks=760*cfg["packets"],per_packet_checks=[190]*cfg["packets"],
            actual_launches=t.expected_launches(cfg),packets=records_by_condition[condition],slot_table=t.slot_table(cfg),
            capture_traces={arm:t.slot_table(cfg)*25 for arm in t.ARMS},samples_ms_per_call=samples,measurements=measurements))
    summary = t.summarize(synthetic)
    assert summary[5]["graph"]["grouped"]["condition_signal"] == "THREE_ROUND_SLOWER_SIGNAL_ONLY"
    assert math.isclose(summary[0]["graph"]["grouped"]["original_over_arm"],1.25)
    assert summary[0]["graph"]["grouped"]["wins_vs_original"] == 3
    reject("missing-worker",lambda:t.summarize(synthetic[:-1]))
    reject("duplicate-worker",lambda:t.summarize(synthetic+[synthetic[0]]))
    mutations = {
        "not-collected":lambda x:x.update(status="FAILED"),
        "lost-packet-check":lambda x:x.update(packet_checks=0),
        "lost-buffer-check":lambda x:x.update(buffer_checks=0),
        "lost-stage":lambda x:x.update(correctness_checks=189),
        "lost-slot":lambda x:x["per_packet_checks"].__setitem__(0,189),
        "wrong-call-count":lambda x:x.update(calls_per_interval=99),
        "wrong-grid":lambda x:x["actual_launches"]["grouped"][0].update(grid=[256]),
        "wrong-warps":lambda x:x["actual_launches"]["frozen_v1"][0]["parameters"].update(num_warps=1),
        "capture-short":lambda x:x["capture_traces"]["original_cuda"].pop(),
        "order-changed":lambda x:x["measurements"].reverse(),
        "sample-short":lambda x:x["samples_ms_per_call"]["grouped"]["graph"].pop(),
        "raw-mismatch":lambda x:x["measurements"][0].update(ms_per_call=123),
    }
    for label,mutate in mutations.items():
        bad = copy.deepcopy(synthetic);mutate(bad[0])
        reject(label,lambda:t.summarize(bad))
    for value in (0,-1,float("nan"),float("inf")):
        bad = copy.deepcopy(synthetic);bad[0]["samples_ms_per_call"]["grouped"]["graph"][0] = value
        reject("invalid-value-"+str(value),lambda:t.summarize(bad))
    # Actual saved35 gate and actual new snapshot, using only a temporary project.
    prior = HERE/"logs/m07-full-groups-timing-results"/t.RUN35
    hashes = json.loads((prior/"source_hashes.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="memory36-cpu-") as tmp:
        project = Path(tmp)/"project"; project.mkdir()
        accepted = project/"experiments/learning/m07-full-groups-timing"/t.RUN35
        accepted.mkdir(parents=True)
        for name in (*hashes,"source_hashes.json","status.json"):
            target = accepted/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(prior/name,target)
        for name in hashes:
            if name.startswith("original_source/"):
                target = project/name.removeprefix("original_source/");target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(prior/name,target)
        for name in (*t.q.PINNED,t.r.GROUPED):
            shutil.copyfile(prior/name,project/"experiments"/name)
        gate = t.accepted_gate(accepted,project)
        run = Path(tmp)/"new36";run.mkdir()
        t.snapshot(project,run);t.verify_snapshot(run,project)
        target = run/"triton_store_grouped.py";good = target.read_bytes();target.write_bytes(good+b"\n# drift\n")
        reject("snapshot-source-drift",lambda:t.verify_snapshot(run,project),RuntimeError);target.write_bytes(good)
        original = project/"python/minisgl/kernel/store.py";good = original.read_bytes();original.write_bytes(good+b"\n# drift\n")
        reject("live-original-drift",lambda:t.verify_snapshot(run,project),RuntimeError);original.write_bytes(good)
        status_path = accepted/"status.json"; good = status_path.read_bytes();state = json.loads(good);state["status"]="FAILED"
        status_path.write_text(json.dumps(state),encoding="utf-8")
        reject("accepted35-failed",lambda:t.accepted_gate(accepted,project));status_path.write_bytes(good)
        live = project/"experiments/triton_store_grouped.py";good = live.read_bytes();live.write_bytes(good+b"\n# drift\n")
        badrun = Path(tmp)/"bad36";badrun.mkdir()
        reject("live-learning-source-drift",lambda:t.snapshot(project,badrun));live.write_bytes(good)
        failed = Path(tmp)/"worker-failure"
        reject("worker-preflight-no-snapshot",lambda:t.worker(project,Path(tmp)/"missing",failed,"A",1),FileNotFoundError)
        failed_result = json.loads((failed/"result.json").read_text(encoding="utf-8"))
        assert failed_result["status"] == "FAILED" and failed_result["error_type"] == "FileNotFoundError"
        assert failed_result["correctness_checks"] == 0 and (failed/"error.txt").exists()
    for name in t.NEW_FILES:
        data = (HERE/name).read_bytes()
        assert not data.startswith(b'\xef\xbb\xbf') and b'\r\n' not in data
        if name.endswith('.py'): ast.parse(data)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    report = dict(status="CPU_MEMORY_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",negative_controls=len(rejected),
        rejected=rejected,fixtures=fixtures,accepted35_gate=gate,
        snapshot="actual temporary new36 snapshot verified including full prior source chain and live original",
        calls="actual raw factory and actual gpu_packet executed with NumPy-backed CUDA-metadata mocks; all arms/all conditions/rotation checked",
        limitations="Mocks and CPU controls do not establish CUDA correctness, capture compatibility or performance",
        source_sha256={name:t.q.sha(HERE/name) for name in t.NEW_FILES})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream: json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],negative_controls=len(rejected),report=str(output),GPU='NOT_RUN')))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
