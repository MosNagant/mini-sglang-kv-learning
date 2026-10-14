"""M07 fixed B256/D128 memory conditions A-F; three frozen implementations.

18 fresh workers; Graph primary, Eager separate; 30 intervals x100 calls.
Offset views keep their complete 1-D backing. F rotates four real packets.
No kernel changes. --dry-run imports neither torch nor triton.
"""
import argparse
import importlib.util
import inspect
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("memory_prior35", HERE/"35-compare-kv-full-groups.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
q,r,h,COMMIT = s.q,s.r,s.h,s.COMMIT
ARMS = ("original_cuda","frozen_v1","grouped")
POLICIES = ("graph","eager")
CASES = tuple("ABCDEF")
B,D,C,SEED = 256,128,1024,20261014
BUFS = ("cache","k","v","indices")
RUN35 = "run-20261010-215026-gs6vimx3"
PINS35 = {"35-compare-kv-full-groups.py":"ae5efd06babab6296038f52e4385bf0295cb998b36a217b03274604f3fb84987",
          "35-compare-kv-full-groups.sh":"624b8fb4cd0db14aca65b747ad102c888cf09e9d32bf91daa679f55c33a47fae"}
HELPERS = tuple(dict.fromkeys((*s.HELPERS,*s.NEW_FILES)))
NEW_FILES = ("36-compare-kv-memory.py","36-compare-kv-memory.sh","36-memory-cpu-test.py")
DESIGN = "reviews/20261014-m07-memory-experiment-design.md"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def configuration(condition):
    require(condition in CASES,"UNPLANNED_MEMORY_CONDITION")
    rows = np.arange(B,dtype=np.int32)
    if condition == "D":
        rows = np.random.default_rng(SEED).permutation(rows)
    return dict(condition=condition,batch=B,width=D,capacity=C,
        layout="separate" if condition == "C" else "interleaved",
        gap=64 if condition == "B" else 0,guard=0,
        source_offset_elements=68 if condition == "E" else 64,
        input_address_mod128=8 if condition == "E" else 0,
        packets=4 if condition == "F" else 1,rows=rows.tolist())


def parameters(cfg,arm):
    require(arm in ARMS[1:],"TRITON_ARM_REQUIRED")
    value = dict(WIDTH=D,SRC_ROW_STRIDE=D+cfg["gap"],
                 DST_ROW_STRIDE=D*(2 if cfg["layout"] == "interleaved" else 1),BLOCK_SIZE=D,num_warps=4)
    if arm == "grouped":
        value.update(BATCH=B,ROWS_PER_PROGRAM=4)
    return value


def grid(arm):
    return (B if arm == "frozen_v1" else B//4,)


def slot_table(cfg):
    return [i % cfg["packets"] for i in range(4)]


def measurement_order(sample,round_id):
    ring = [(arm,policy) for policy in POLICIES for arm in ARMS]
    offset = (sample+round_id-1) % 6
    order = ring[offset:]+ring[:offset]
    return list(reversed(order)) if (sample//6+round_id-1) % 2 else order


def make_plan():
    rng,schedule = random.Random(SEED),[]
    for round_id in (1,2,3):
        conditions = list(CASES)
        rng.shuffle(conditions)
        schedule.extend(dict(round=round_id,condition=x) for x in conditions)
    return dict(configurations=[configuration(x) for x in CASES],schedule=schedule,
        implementations=list(ARMS),policies=list(POLICIES),fresh_processes=18,
        intervals_per_arm_policy=30,calls_per_interval=100,interval_averages=3240,
        seed=SEED,primary="graph",reference="eager",eager_warmup_calls=20,graph_warmup_replays=5,
        address_policy="A-E four-entry selection table aliases one packet; F cycles four disjoint packets,25 calls each per interval; no cache flush",
        source_policy="same frozen v1/grouped and original CUDA; no kernel mutation",
        correctness="every unique packet: initial per-arm call, reset Graph replay, warmup, every measured interval and final full backing",
        expected_check_stages=190,packet_checks="190 multiplied by unique packets; each stage compares all four complete buffers per packet",
        exclusion="allocation, metadata/index validation, first JIT, warmup, resets, checks, compiler export and logging outside timed intervals",
        timed_entries="original public store_cache vs trusted raw Triton launches; all Eager arms include identical four-item Python selection mechanism",
        statistic="median of30 interval averages per process, then median of3 process medians; paired arm ratios separately",
        signal_rule="all3 per-round condition/A time ratios >=1.03 or <=0.97; engineering signal only, no significance or automatic promotion",
        comparison_limit="arms paired in same worker; condition/A paired only by round across different processes",
        ownership="user authorized autonomous coach implementation and execution on2026-10-14; learner mastery assessed separately",
        scope="RTX4070Ti FP16 B256 D128 C1024; fixed slot set; offset8B remains naturally8B-aligned; rotation is about1MiB touched data, not a cold-cache/DRAM/model/AMD result")


def fixture(cfg,slot):
    require(cfg == configuration(cfg["condition"]) and slot in range(cfg["packets"]),"FIXTURE_SCOPE")
    stride,offset = D+cfg["gap"],cfg["source_offset_elements"]
    shape = (C,2,D) if cfg["layout"] == "interleaved" else (2,C,D)
    before = dict(cache=np.full(shape,-1,np.float16),
        k=np.full(64+B*stride+68,-9,np.float16),v=np.full(64+B*stride+68,-9,np.float16),
        indices=np.array(cfg["rows"],np.int32))
    values = np.random.default_rng(SEED+101*slot).integers(0,1024,(B,D),dtype=np.int32).astype(np.float16)
    values[:,0] = np.arange(B,dtype=np.float16)+np.float16(slot*B)
    expected = before["cache"].copy()
    for row,position in enumerate(before["indices"]):
        begin = offset+row*stride
        before["k"][begin:begin+D] = values[row]
        before["v"][begin:begin+D] = values[row]+np.float16(1024)
        if cfg["layout"] == "interleaved":
            expected[position,0,:],expected[position,1,:] = before["k"][begin:begin+D],before["v"][begin:begin+D]
        else:
            expected[0,position,:],expected[1,position,:] = before["k"][begin:begin+D],before["v"][begin:begin+D]
    before["expected_cache"] = expected
    return before


def verify_arrays(cfg,slot,before,after):
    reference = fixture(cfg,slot)
    require(set(before) == set(reference) and set(after) == set(BUFS),"ARRAY_KEYS_CHANGED")
    for name in reference:
        require(before[name].dtype == reference[name].dtype and np.array_equal(before[name],reference[name]),"FIXTURE_CHANGED: "+name)
    for name in BUFS:
        target = reference["expected_cache" if name == "cache" else name]
        require(after[name].dtype == target.dtype and np.array_equal(after[name],target),"BACKING_MISMATCH: "+name)


def tensor_record(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),
                data_ptr=t.data_ptr(),storage_offset=t.storage_offset(),bytes=t.numel()*t.element_size())


def gpu_packet(torch,cfg,before):
    buffers = {name:torch.from_numpy(before[name].copy()).cuda() for name in BUFS}
    kc,vc = h.cache_views(buffers["cache"],cfg["layout"])
    source = [buffers[name].as_strided((B,D),(D+cfg["gap"],1),cfg["source_offset_elements"]) for name in ("k","v")]
    args = (kc,vc,buffers["indices"],*source)
    return buffers,args


def validate_address_records(cfg,records):
    require(len(records) == cfg["packets"],"PACKET_COUNT")
    occupied = []
    for slot,record in enumerate(records):
        require(record["slot"] == slot,"PACKET_ORDER")
        bufs,views = record["buffers"],record["views"]
        require(set(bufs) == set(BUFS) and len(views) == 5,"PACKET_RECORD_KEYS")
        devices = {x["device"] for x in [*bufs.values(),*views]}
        require(len(devices) == 1 and next(iter(devices)).startswith("cuda"),"PACKET_DEVICE")
        before = fixture(cfg,slot)
        for name in BUFS:
            item = bufs[name]
            dtype = "torch.int32" if name == "indices" else "torch.float16"
            expected_stride = [x//before[name].itemsize for x in before[name].strides]
            require(item["shape"] == list(before[name].shape) and item["stride"] == expected_stride and
                    item["dtype"] == dtype and item["bytes"] == before[name].nbytes and item["storage_offset"] == 0,
                    "BACKING_METADATA: "+name)
            require(type(item["data_ptr"]) is int and item["data_ptr"] > 0 and item["data_ptr"] % 128 == 0,"BACKING_ALIGNMENT")
            occupied.append((item["data_ptr"],item["data_ptr"]+item["bytes"]))
        cache_v_offset = D if cfg["layout"] == "interleaved" else C*D
        specs = [("cache",0,[C,D],[parameters(cfg,"grouped")["DST_ROW_STRIDE"],1]),
                 ("cache",cache_v_offset,[C,D],[parameters(cfg,"grouped")["DST_ROW_STRIDE"],1]),
                 ("indices",0,[B],[1]),
                 ("k",cfg["source_offset_elements"],[B,D],[D+cfg["gap"],1]),
                 ("v",cfg["source_offset_elements"],[B,D],[D+cfg["gap"],1])]
        for item,(name,offset,shape,stride) in zip(views,specs):
            require(item["shape"] == shape and item["stride"] == stride and item["storage_offset"] == offset and
                    item["data_ptr"] == bufs[name]["data_ptr"]+offset*(4 if name == "indices" else 2) and
                    item["dtype"] == bufs[name]["dtype"] and item["bytes"] == math.prod(shape)*(4 if name == "indices" else 2),
                    "VIEW_METADATA_OR_OFFSET")
            require(item["data_ptr"] % (4 if name == "indices" else 8) == 0,"ORIGINAL_PACKAGE_ALIGNMENT")
        require(all(x["data_ptr"] % 128 == cfg["input_address_mod128"] for x in views[3:]),"INPUT_ALIGNMENT_CONDITION_LOST")
    occupied.sort()
    require(all(a[1] <= b[0] for a,b in zip(occupied,occupied[1:])),"PACKET_BACKING_ALIAS")


def raw_call(original,kernel,cfg,arm,args):
    if arm == "original_cuda":
        def call():
            return original(*args)
    else:
        launch_grid,kwargs = grid(arm),parameters(cfg,arm)
        def call():
            return kernel[launch_grid](*args,**kwargs)
    return call


class SlotCycle:
    """Same four-entry host selector in every condition; no validation in timed path."""
    def __init__(self,calls,cfg):
        self.slot_ids = slot_table(cfg)
        self.table = tuple(calls[i] for i in self.slot_ids)
        self.reset()

    def reset(self):
        self.position = 0

    def __call__(self):
        position = self.position
        self.position = (position+1) & 3
        return self.table[position]()


class CaptureCycle(SlotCycle):
    """Records host submissions during capture only; graph replay contains raw GPU work."""
    def __init__(self,calls,cfg):
        super().__init__(calls,cfg)
        self.trace = []

    def __call__(self):
        self.trace.append(self.slot_ids[self.position])
        return super().__call__()


def expected_launches(cfg):
    result = {arm:[] for arm in ARMS}
    for arm in ARMS:
        for slot in range(cfg["packets"]):
            item = dict(slot=slot)
            if arm == "original_cuda":
                item.update(entry="upstream store_cache",public_calls=1)
            else:
                item.update(grid=list(grid(arm)),parameters=parameters(cfg,arm))
            result[arm].append(item)
    return result


def signal(ratios):
    if all(x >= 1.03 for x in ratios):
        return "THREE_ROUND_SLOWER_SIGNAL_ONLY"
    if all(x <= .97 for x in ratios):
        return "THREE_ROUND_FASTER_SIGNAL_ONLY"
    return "SMALL_OR_INCONSISTENT_NO_PROMOTION"


def summarize(results):
    keyed = {}
    expected_keys = {(round_id,condition) for round_id in (1,2,3) for condition in CASES}
    for result in results:
        key = (result["round"],result["condition"])
        require(key in expected_keys and key not in keyed and result.get("status") == "COLLECTED","RESULT_ID_OR_STATUS")
        cfg = configuration(key[1])
        require(result.get("config") == cfg and result.get("calls_per_interval") == 100,"RESULT_CONFIGURATION")
        require(result.get("correctness_checks") == 190 and result.get("packet_checks") == 190*cfg["packets"] and
                result.get("buffer_checks") == 760*cfg["packets"] and
                result.get("per_packet_checks") == [190]*cfg["packets"],"INCOMPLETE_BACKING_CHECKS")
        require(result.get("actual_launches") == expected_launches(cfg),"ACTUAL_LAUNCH_MISMATCH")
        validate_address_records(cfg,result["packets"])
        require(result.get("slot_table") == slot_table(cfg),"SLOT_TABLE_CHANGED")
        trace = slot_table(cfg)*25
        require(result.get("capture_traces") == {arm:trace for arm in ARMS},"CAPTURE_SEQUENCE_MISMATCH")
        records = result["measurements"]
        order = [(sample,arm,policy) for sample in range(30) for arm,policy in measurement_order(sample,key[0])]
        require([(x["sample"],x["implementation"],x["policy"]) for x in records] == order,"MEASUREMENT_ORDER")
        for arm in ARMS:
            for policy in POLICIES:
                values = result["samples_ms_per_call"][arm][policy]
                require(len(values) == 30 and all(math.isfinite(x) and x > 0 for x in values),"INVALID_TIMING_SAMPLE")
                require(values == [x["ms_per_call"] for x in records if (x["implementation"],x["policy"]) == (arm,policy)],"RAW_SAMPLE_MISMATCH")
        keyed[key] = result
    require(set(keyed) == expected_keys,"INCOMPLETE_18_WORKER_MATRIX")
    summary = []
    for condition in CASES:
        row = dict(condition=condition,config=configuration(condition))
        for policy in POLICIES:
            entry = {}
            for arm in ARMS:
                medians = [statistics.median(keyed[(round_id,condition)]["samples_ms_per_call"][arm][policy])*1000 for round_id in (1,2,3)]
                baseline = [statistics.median(keyed[(round_id,"A")]["samples_ms_per_call"][arm][policy])*1000 for round_id in (1,2,3)]
                ratios = [a/b for a,b in zip(medians,baseline)]
                entry[arm] = dict(round_medians_us=medians,representative_us=statistics.median(medians),
                    round_span_pct=100*(max(medians)-min(medians))/statistics.median(medians),
                    condition_over_A_round_time_ratios=ratios,condition_signal=signal(ratios) if condition != "A" else "BASELINE")
            for arm in ARMS[1:]:
                ratios = [a/b for a,b in zip(entry["original_cuda"]["round_medians_us"],entry[arm]["round_medians_us"])]
                entry[arm].update(original_over_arm=entry["original_cuda"]["representative_us"]/entry[arm]["representative_us"],
                    paired_original_over_arm=ratios,paired_ratio_median=statistics.median(ratios),wins_vs_original=sum(x>1 for x in ratios))
            row[policy] = entry
        summary.append(row)
    return summary


def accepted_gate(directory,project):
    s.verify_snapshot(directory,project)
    state = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    expected_plan = json.loads(json.dumps(s.make_plan()))
    require(state.get("status") == "COLLECTION_COMPLETE" and state.get("commit") == COMMIT and state.get("plan") == expected_plan,"ACCEPTED35_REQUIRED")
    require(state["summary"] == s.summarize(state["results"]),"ACCEPTED35_RECORDS_CHANGED")
    hashes = json.loads((directory/"source_hashes.json").read_text(encoding="utf-8"))
    for name in HELPERS:
        require(q.sha(HERE/name) == hashes.get(name),"EXECUTED_HELPER_DRIFT: "+name)
    for name,expected in PINS35.items():
        require(q.sha(directory/name) == expected,"ACCEPTED35_RUNNER_CHANGED")
    return dict(status="ACCEPTED35_SOURCE_AND_SAVED_RECORDS_PASS",run=RUN35,
        status_sha256=q.sha(directory/"status.json"),source_hashes_sha256=q.sha(directory/"source_hashes.json"),
        limitation="saved sources/statistics checked, not rerunning old GPU arrays; all36 actual input packets checked separately")


def verify_snapshot(run,project):
    hashes = json.loads((run/"source_hashes.json").read_text(encoding="utf-8"))
    required = (*HELPERS,*NEW_FILES,*q.PINNED,r.GROUPED,"hypothesis-card.md","accepted35-gate.json", "prior35-status.json","prior35-source_hashes.json")
    require(all(name in hashes for name in required),"INCOMPLETE_MEMORY_SNAPSHOT")
    s.verify_snapshot(run,project)
    for name,expected in PINS35.items():
        require(q.sha(run/name) == expected,"ACCEPTED35_RUNNER_CHANGED")
    q.check_pinned(run)
    require(q.sha(run/r.GROUPED) == s.p.GROUPED_HASH,"GROUPED_SOURCE_CHANGED")
    r.review_source((run/r.GROUPED).read_text(encoding="utf-8-sig"))


def snapshot(project,run):
    for name in (*HELPERS,*NEW_FILES):
        shutil.copyfile(HERE/name,run/name)
    for name in (*q.PINNED,r.GROUPED):
        shutil.copyfile(project/"experiments"/name,run/name)
    h.write_json(run/"input_source_hashes.json",{x.name:q.sha(x) for x in run.iterdir() if x.is_file()})
    prior = project/"experiments/learning/m07-full-groups-timing"/RUN35
    gate = accepted_gate(prior,project)
    prior_hashes = json.loads((prior/"source_hashes.json").read_text(encoding="utf-8"))
    for name in (*q.PINNED,r.GROUPED):
        require(q.sha(run/name) == prior_hashes.get(name),"LIVE_LEARNING_SOURCE_DRIFT: "+name)
    for name in prior_hashes:
        target = run/name
        if name == "hypothesis-card.md":
            target = run/"prior35-hypothesis-card.md"
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(prior/name,target)
    for name in ("status.json","source_hashes.json"):
        shutil.copyfile(prior/name,run/("prior35-"+name))
    h.write_json(run/"accepted35-gate.json",gate)
    shutil.copyfile(HERE/DESIGN,run/"hypothesis-card.md")
    h.write_json(run/"source_hashes.json",{x.relative_to(run).as_posix():q.sha(x) for x in run.rglob("*") if x.is_file()})
    verify_snapshot(run,project)


def worker(project,run,directory,condition,round_id):
    directory.mkdir(parents=True)
    result = dict(status="STARTED",condition=condition,round=round_id,calls_per_interval=100,
        correctness_checks=0,packet_checks=0,buffer_checks=0,
        samples_ms_per_call={arm:{policy:[] for policy in POLICIES} for arm in ARMS},measurements=[])
    h.write_json(directory/"result.json",result)
    try:
        cfg = configuration(condition)
        require(round_id in (1,2,3),"UNPLANNED_ROUND")
        result.update(config=cfg,slot_table=slot_table(cfg),per_packet_checks=[0]*cfg["packets"])
        verify_snapshot(run,project)
        eager_timer = s.load_reviewed_timer(run/"learner-timing.py")
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        require(torch.cuda.is_available(),"CUDA_NOT_AVAILABLE")
        require((torch.__version__,torch.version.cuda,triton.__version__) == ("2.9.1+cu128","12.8","3.5.1"),"FIXED_ENVIRONMENT_CHANGED")
        require(Path(inspect.getsourcefile(store_cache)).resolve() == (project/"python/minisgl/kernel/store.py").resolve(),"WRONG_MINISGL_IMPORT")
        support = h.load_module(run/"kv_contract_support.py","kv_contract_support")
        kernel_v1 = h.load_module(run/"triton_store_mask.py","triton_store_mask").store_kv_v1
        grouped = h.load_module(run/r.GROUPED,"memory36_grouped")
        require((grouped.NUM_WARPS,grouped.ROWS_PER_PROGRAM) == (4,4),"GROUP_PARAMETERS_CHANGED")
        kernels = dict(frozen_v1=kernel_v1,grouped=grouped.store_kv_grouped)
        result.update(commit=COMMIT,torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,
            gpu=torch.cuda.get_device_name(0),device_properties=str(torch.cuda.get_device_properties(0)),telemetry_before=s.telemetry())
        packets,records = [],[]
        for slot in range(cfg["packets"]):
            path = directory/f"packet-{slot}"
            path.mkdir()
            before = fixture(cfg,slot)
            np.savez_compressed(path/"before.npz",**before)
            buffers,args = gpu_packet(torch,cfg,before)
            expected = {name:torch.from_numpy(before["expected_cache" if name == "cache" else name].copy()).cuda() for name in BUFS}
            packets.append(dict(path=path,buffers=buffers,args=args,expected=expected,
                                initial_cache=torch.from_numpy(before["cache"].copy()).cuda()))
            records.append(dict(slot=slot,buffers={name:tensor_record(t) for name,t in buffers.items()},views=[tensor_record(t) for t in args]))
            b,d,src,dst,block = support.validate_metadata(*args)
            require((b,d,src,dst,block) == (B,D,D+cfg["gap"],parameters(cfg,"grouped")["DST_ROW_STRIDE"],D),"ACTUAL_LAYOUT_MISMATCH")
            support.validate_index_values(args[2],B,C)
        result["packets"] = records
        h.write_json(directory/"packets.json",records)
        validate_address_records(cfg,records)

        def save_arrays(stage):
            for packet in packets:
                np.savez_compressed(packet["path"]/(stage+".npz"),**{name:t.cpu().numpy().copy() for name,t in packet["buffers"].items()})

        def reset_caches():
            for packet in packets:
                packet["buffers"]["cache"].copy_(packet["initial_cache"])

        def check(stage,save=False):
            try:
                for slot,packet in enumerate(packets):
                    for name in BUFS:
                        require(torch.equal(packet["buffers"][name],packet["expected"][name]),f"BACKING_MISMATCH {stage} slot={slot} {name}")
                        result["buffer_checks"] += 1
                    result["packet_checks"] += 1
                    result["per_packet_checks"][slot] += 1
                result["correctness_checks"] += 1
            except BaseException:
                try:
                    save_arrays("failed-"+stage)
                except Exception as diagnostic:
                    result.setdefault("failure_capture_errors",[]).append(dict(stage=stage,error=str(diagnostic)))
                raise
            if save:
                save_arrays(stage)

        raw = {arm:[raw_call(store_cache,kernels.get(arm),cfg,arm,packet["args"]) for packet in packets] for arm in ARMS}
        calls = {arm:SlotCycle(raw[arm],cfg) for arm in ARMS}
        graphs = {}
        offset = (round_id-1) % 3
        result.update(setup_order=list(ARMS[offset:]+ARMS[:offset]),actual_launches={arm:[] for arm in ARMS},capture_traces={})
        for arm in result["setup_order"]:
            reset_caches()
            compiled = None
            for slot,packet in enumerate(packets):
                if arm == "original_cuda":
                    raw[arm][slot]()
                    entry = dict(slot=slot,entry="upstream store_cache",public_calls=1)
                else:
                    probe = q.LaunchProbe(kernels[arm],4) if arm == "frozen_v1" else r.GroupedProbe(kernels[arm])
                    probe.reset(True)
                    returned = probe[grid(arm)](*packet["args"],**parameters(cfg,arm))
                    require(probe.attempts == [dict(grid=list(grid(arm)),parameters=parameters(cfg,arm))],"INITIAL_LAUNCH_PROBE")
                    entry = dict(slot=slot,**probe.attempts[0])
                    if slot == 0:
                        compiled = returned
                result["actual_launches"][arm].append(entry)
            torch.cuda.synchronize()
            check(arm+"-initial",save=True)
            if arm != "original_cuda":
                artifact = directory/(arm+"-first-packet")
                artifact.mkdir()
                s.v.e.export_compiled(compiled,artifact)
            call = calls[arm]
            call.reset()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(20):
                    call()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            capture_call = CaptureCycle(raw[arm],cfg)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=side):
                for _ in range(100):
                    capture_call()
            require(capture_call.trace == slot_table(cfg)*25,"ACTUAL_CAPTURE_SLOT_SEQUENCE")
            result["capture_traces"][arm] = capture_call.trace
            reset_caches()
            graph.replay()
            torch.cuda.synchronize()
            check(arm+"-graph-replay",save=True)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            check(arm+"-warmup")
            graphs[arm] = graph
        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record(); end.record(); end.synchronize()
        print(f"MEMORY_READY condition={condition} round={round_id} packets={cfg['packets']} full backing checked",flush=True)
        for sample in range(30):
            for arm,policy in measurement_order(sample,round_id):
                calls[arm].reset()
                ms = float(eager_timer(calls[arm],start,end) if policy == "eager" else s.measure_graph(graphs[arm],start,end))
                require(math.isfinite(ms) and ms > 0,"INVALID_TIMING")
                require(calls[arm].position == 0,"EAGER_SLOT_CYCLE_INCOMPLETE")
                result["samples_ms_per_call"][arm][policy].append(ms)
                result["measurements"].append(dict(sample=sample,implementation=arm,policy=policy,ms_per_call=ms))
                check(f"sample-{sample}-{arm}-{policy}")
            h.write_json(directory/"result.json",result)
        check("final",save=True)
        result["telemetry_after"] = s.telemetry()
        verify_snapshot(run,project)
        require(result["actual_launches"] == expected_launches(cfg),"INITIAL_SUBMISSION_RECORDS")
        result.update(status="COLLECTED",policy=make_plan(),scope="actual packet correctness passed; timings require review")
    except BaseException as error:
        result.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (directory/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(directory/"result.json",result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path)
    parser.add_argument("--shared-results",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--worker-dir",type=Path)
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--condition",choices=CASES)
    parser.add_argument("--round",type=int,choices=(1,2,3))
    args = parser.parse_args()
    require(not sys.flags.optimize,"ASSERTS_REQUIRED")
    if args.dry_run:
        print(json.dumps(make_plan(),indent=2)); return
    if args.project is None:
        parser.error("--project required")
    project = args.project.resolve()
    if args.worker_dir:
        if None in (args.snapshot,args.condition,args.round):
            parser.error("worker requires snapshot/condition/round")
        worker(project,args.snapshot,args.worker_dir,args.condition,args.round); return
    if args.shared_results is None:
        parser.error("--shared-results required")
    def git(*words):
        return subprocess.check_output(["git","-C",str(project),*words],text=True).strip()
    parent = project/"experiments/learning/m07-memory"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 controlled memory conditions",commit=COMMIT,plan=make_plan(),results=[],ownership=make_plan()["ownership"])
    print(f"RUN_DIRECTORY={run}",flush=True)
    try:
        require(git("rev-parse","HEAD") == COMMIT and not git("status","--porcelain","--untracked-files=no"),"PINNED_TRACKED_SOURCE_CHANGED")
        snapshot(project,run)
        h.write_json(run/"plan.json",state["plan"])
        identity = None
        for item in state["plan"]["schedule"]:
            condition,round_id = item["condition"],item["round"]
            directory = run/f"round-{round_id}-{condition}"
            console = run/f"round-{round_id}-{condition}.log"
            command = [sys.executable,"-B",str(run/NEW_FILES[0]),"--project",str(project),"--snapshot",str(run),
                "--worker-dir",str(directory),"--condition",condition,"--round",str(round_id)]
            state.update(status="RUNNING",current=item)
            h.write_json(run/"status.json",state)
            h.write_json(run/(console.stem+"-command.json"),command)
            print(f"RUN round={round_id}/3 condition={condition}",flush=True)
            with console.open("x",encoding="utf-8") as log:
                completed = subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                print(console.read_text(encoding="utf-8"),flush=True)
                raise RuntimeError("MEMORY_WORKER_FAILED: "+str(console))
            result = json.loads((directory/"result.json").read_text(encoding="utf-8"))
            require(result["status"] == "COLLECTED" and all(result[k] == value for k,value in item.items()),"WORKER_RESULT_MISMATCH")
            environment = tuple(result[k] for k in ("commit","torch","cuda","triton","gpu","device_properties"))
            if identity is None:
                identity = environment
            require(identity == environment,"ENVIRONMENT_CHANGED_BETWEEN_WORKERS")
            state["results"].append(result)
            h.write_json(run/"partial-results.json",state["results"])
            for policy in POLICIES:
                values = {arm:statistics.median(result["samples_ms_per_call"][arm][policy])*1000 for arm in ARMS}
                print("COLLECTED "+condition+" "+policy+" "+json.dumps(values)+" us/call",flush=True)
        require(git("rev-parse","HEAD") == COMMIT and not git("status","--porcelain","--untracked-files=no"),"TRACKED_SOURCE_CHANGED_DURING_RUN")
        verify_snapshot(run,project)
        summary = summarize(state["results"])
        state.update(status="COLLECTION_COMPLETE",current=None,summary=summary)
        h.write_json(run/"summary.json",dict(status=state["status"],plan=state["plan"],summary=summary))
        for row in summary:
            print("GRAPH_SUMMARY condition="+row["condition"]+" "+json.dumps(row["graph"]),flush=True)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run/"status.json",state)
        s.export_run(run,args.shared_results,state["status"] == "COLLECTION_COMPLETE")
    print("MEMORY_COLLECTION_COMPLETE:18 workers;3240 interval averages; engineering signals only; no automatic promotion")


if __name__ == "__main__":
    main()
