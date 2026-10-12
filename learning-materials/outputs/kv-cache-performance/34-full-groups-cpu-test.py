"""CPU-only source/template/fixture/probe controls for34; no Torch/WSL/GPU."""
import ast
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("full_groups_under_test",HERE / "34-check-kv-full-groups.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def main(output):
    if output.exists():
        raise FileExistsError("REPORT_EXISTS: choose a new output")
    rejected = []
    def reject(label,call,error=RuntimeError):
        try:
            call()
        except error:
            rejected.append(label)
        else:
            raise AssertionError("NEGATIVE_CONTROL_NOT_REJECTED: " + label)
    accepted = HERE / "logs/m07-grouped-six-timing-results" / t.ACCEPTED_TIMING_RUN
    grouped = (accepted/t.r.GROUPED).read_text(encoding="utf-8-sig")
    template = t.template_source(grouped)
    assert "ONE if condition" in template and "THREE expressions" not in template
    assert sum(isinstance(n,ast.Constant) and n.value is Ellipsis for n in ast.walk(ast.parse(template))) == 1
    reference = template.replace("if ...:","if BATCH % ROWS_PER_PROGRAM == 0:",1)
    t.review_source(reference,grouped)
    t.review_source(ast.unparse(ast.parse(reference))+"\n# formatting/comment allowed\n",grouped)
    changes = {
        "unfilled":template,
        "always_true_tail_risk":reference.replace("if BATCH % ROWS_PER_PROGRAM == 0:","if True:"),
        "always_false_not_specialized":reference.replace("if BATCH % ROWS_PER_PROGRAM == 0:","if False:"),
        "inverted_full_condition":reference.replace("% ROWS_PER_PROGRAM == 0","% ROWS_PER_PROGRAM != 0"),
        "size_not_multiple":reference.replace("BATCH % ROWS_PER_PROGRAM == 0","BATCH >= ROWS_PER_PROGRAM"),
        "scalar_true_mask":reference.replace("tl.full((ROWS_PER_PROGRAM,), True, tl.int1)","True"),
        "wrong_full_shape":reference.replace("tl.full((ROWS_PER_PROGRAM,), True, tl.int1)","tl.full((), True, tl.int1)"),
        "wrong_full_dtype":reference.replace("True, tl.int1","True, tl.int32"),
        "wrong_tail_predicate":reference.replace("rows < BATCH","rows <= BATCH"),
        "index_mask_removed":reference.replace("mask=row_mask, other=0","other=0"),
        "combined_row_mask_removed":reference.replace("row_mask[:, None] & col_mask[None, :]","col_mask[None, :]"),
        "column_mask_removed":reference.replace("cols < WIDTH","cols < BLOCK_SIZE"),
        "constexpr_removed":reference.replace("BATCH: tl.constexpr","BATCH"),
        "warps_changed":reference.replace("NUM_WARPS = 4","NUM_WARPS = 1"),
        "rows_per_program_changed":reference.replace("ROWS_PER_PROGRAM = 4","ROWS_PER_PROGRAM = 8"),
        "floor_grid":reference.replace("triton.cdiv(batch, ROWS_PER_PROGRAM)","batch // ROWS_PER_PROGRAM"),
        "source_stride_changed":reference.replace("rows[:, None] * SRC_ROW_STRIDE","rows[:, None] * WIDTH"),
        "destination_stride_changed":reference.replace("positions[:, None] * DST_ROW_STRIDE","positions[:, None] * WIDTH"),
    }
    for name,source in changes.items():
        reject(name,lambda:t.review_source(source,grouped))
    configs = t.configurations()
    assert len(configs) == 23 and sum(x["batch"] > 0 for x in configs) == 21
    assert {(x["batch"],x["width"]) for x in configs if x["batch"]} == {
        (b,d) for b in (1,3,4,5,8,17,256) for d in (64,128,192)}
    assert sum(x["batch"] > 0 and x["batch"]%4 == 0 for x in configs) == 9
    assert sum(t.selected_compile(x) for x in configs) == 2
    condition = next(n.test for n in ast.walk(ast.parse(reference)) if isinstance(n,ast.If)
                     and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.BinOp))
    expression = compile(ast.Expression(condition),"reviewed-constexpr-condition","eval")
    coverage = []
    class Tensor:
        def __init__(self,shape,stride): self.shape,self.row_stride = shape,stride
        def stride(self,axis):
            assert axis == 0
            return self.row_stride
    class Kernel:
        def __init__(self): self.calls,self.compiled = [],object()
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                self.calls.append(dict(grid=list(grid),parameters=kwargs))
                return self.compiled
            return call
    for cfg in configs:
        b,d = cfg["batch"],cfg["width"]
        assert len(set(cfg["rows"])) == b and all(0 <= x < 512 for x in cfg["rows"])
        before = t.fixture(cfg)
        good = t.c.correct_after(before)
        t.h.verify_backing(before,good)
        assert all(np.isfinite(a).all() for a in before.values())
        assert np.array_equal(before["k"][:,0],np.arange(b,dtype=np.float16))
        full = eval(expression,{"BATCH":b,"ROWS_PER_PROGRAM":4})
        visited = []
        for pid in range((b+3)//4):
            rows = pid*4+np.arange(4)
            row_mask = np.ones(4,dtype=bool) if full else rows < b
            assert np.array_equal(row_mask,rows < b)
            visited.extend(rows[row_mask].tolist())
            cols = np.arange(1 << (d-1).bit_length())
            valid = row_mask[:,None] & (cols[None,:] < d)
            rr,cc = np.where(valid)
            assert np.all(rows[rr] < b) and np.all(cols[cc] < d)
        assert visited == list(range(b))
        src,dst = d+cfg["gap"],(d+cfg["guard"])*(2 if cfg["layout"] == "interleaved" else 1)
        args = [Tensor((512,d),dst),Tensor((512,d),dst),Tensor((b,),1),Tensor((b,d),src),Tensor((b,d),src)]
        kwargs = dict(BATCH=b,WIDTH=d,SRC_ROW_STRIDE=src,DST_ROW_STRIDE=dst,
                      BLOCK_SIZE=1 << (d-1).bit_length(),ROWS_PER_PROGRAM=4,num_warps=4)
        kernel = Kernel()
        probe = t.CompiledCapture(kernel)
        probe.reset(b > 0)
        grid = ((b+3)//4,)
        if b:
            assert probe[grid](*args,**kwargs) is kernel.compiled and probe.compiled is kernel.compiled
            assert probe.attempts == kernel.calls
        else:
            reject(f"B0-D{d}-launch-blocked",lambda:probe[grid](*args,**kwargs))
            assert not kernel.calls
        calls = len(kernel.calls)
        for name,bad_grid,bad_params in (("grid",(grid[0]+1,),kwargs),
            ("stride",grid,dict(kwargs,SRC_ROW_STRIDE=src+1)),("warps",grid,dict(kwargs,num_warps=1))):
            probe.reset(b > 0)
            assert probe.compiled is None and probe.attempts == []
            reject(f"B{b}-D{d}-{name}",lambda:probe[bad_grid](*args,**bad_params))
            assert len(kernel.calls) == calls and probe.compiled is None
        coverage.append(dict(batch=b,width=d,full_condition=bool(full),grid=list(grid),source_stride=src,destination_stride=dst))
    # Evaluate the rejected source's condition: B5 would permit three invalid row addresses.
    bad_tree = ast.parse(changes["always_true_tail_risk"])
    bad_condition = next(n.test for n in ast.walk(bad_tree) if isinstance(n,ast.If)
                         and isinstance(n.test,ast.Constant) and n.test.value is True)
    bad_full = eval(compile(ast.Expression(bad_condition),"rejected-tail-risk-condition","eval"),{})
    invalid_rows = []
    for pid in range((5+3)//4):
        rows = pid*4+np.arange(4)
        permitted = np.ones(4,dtype=bool) if bad_full else rows < 5
        invalid_rows.extend(rows[permitted & (rows >= 5)].tolist())
    assert invalid_rows == [5,6,7]
    for b,d in ((5,192),(256,128)):
        cfg = next(x for x in configs if (x["batch"],x["width"]) == (b,d))
        before,good = t.fixture(cfg),t.c.correct_after(t.fixture(cfg))
        unused = next(x for x in range(512) if x not in cfg["rows"])
        targets = [("targetK","cache",(0,cfg["rows"][-1],d-1)),("targetV","cache",(1,cfg["rows"][-1],d-1)),
            ("untouched","cache",(0,unused,0)),("inputK","k",(b-1,d-1)),("inputV","v",(b-1,d-1)),
            ("input_gap","k",(b-1,d)),("indices","indices",b-1)]
        if cfg["guard"]:
            targets.append(("cache_guard","cache",(1,511,d)))
        for name,array,index in targets:
            bad = {n:a.copy() for n,a in good.items()}
            bad[array][index] = -99
            reject(f"fixture-B{b}-D{d}-{name}",lambda:t.h.verify_backing(before,bad),AssertionError)
        bad = {n:a.copy() for n,a in good.items()}
        first,second = cfg["rows"][0],cfg["rows"][-1]
        bad["cache"][:,[first,second]] = bad["cache"][:,[second,first]]
        reject(f"fixture-B{b}-D{d}-row-swap",lambda:t.h.verify_backing(before,bad),AssertionError)
    reject("duplicate-indices-configuration",lambda:t.fixture(dict(configs[0],rows=[11,11])),ValueError)
    reject("out-of-range-configuration",lambda:t.fixture(dict(configs[0],rows=[512])),ValueError)
    with tempfile.TemporaryDirectory(prefix="m07-full-groups-cpu-") as temporary:
        project = Path(temporary)
        exp = project / "experiments"
        exp.mkdir()
        for name in (*t.q.PINNED,t.r.GROUPED):
            shutil.copyfile(accepted/name,exp/name)
        before_hashes = {name:t.q.sha(exp/name) for name in (*t.q.PINNED,t.r.GROUPED)}
        destination = t.prepare(project)
        assert destination.read_text(encoding="utf-8") == template
        reject("prepared-file-still-unfilled",lambda:t.review_source(destination.read_text(encoding="utf-8"),grouped))
        destination.write_text(reference,encoding="utf-8",newline="\n")
        filled = destination.read_bytes()
        assert t.prepare(project).read_bytes() == filled
        assert before_hashes == {name:t.q.sha(exp/name) for name in before_hashes}
        # Exercise the new provenance/archiving path using only old saved source files.
        archived = project / "experiments/learning/m07-grouped-six-timing" / t.ACCEPTED_TIMING_RUN
        archived.mkdir(parents=True)
        old_hashes = json.loads((accepted/"source_hashes.json").read_text(encoding="utf-8"))
        for name in (*old_hashes,"source_hashes.json","status.json"):
            target = archived/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(accepted/name,target)
            if name.startswith("original_source/"):
                live = project/name.removeprefix("original_source/")
                live.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(accepted/name,live)
        run = project/"snapshot"
        run.mkdir()
        t.snapshot(project,run)
        t.verify_snapshot(run)
        (run/t.CANDIDATE).write_text("changed",encoding="utf-8")
        reject("candidate-snapshot-change",lambda:t.verify_snapshot(run))
        destination.write_text(template,encoding="utf-8",newline="\n")
        failed_run = project/"unfilled-snapshot"
        failed_run.mkdir()
        reject("unfilled-source-stops-before-GPU",lambda:t.snapshot(project,failed_run))
        assert (failed_run/t.CANDIDATE).read_text(encoding="utf-8") == template
        assert json.loads((failed_run/"source_hashes.json").read_text(encoding="utf-8"))[t.CANDIDATE] == t.q.sha(failed_run/t.CANDIDATE)
        destination.write_bytes(filled)
        (exp/t.r.GROUPED).write_text("changed",encoding="utf-8")
        reject("changed-baseline-blocks-prepare",lambda:t.prepare(project))
        assert destination.read_bytes() == filled
    for name in t.NEW_FILES:
        raw = (HERE/name).read_bytes()
        assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
        if name.endswith(".py"):
            ast.parse(raw.decode("utf-8"))
    assert "torch" not in sys.modules and "triton" not in sys.modules
    report = dict(status="CPU_FULL_GROUPS_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",plan=t.make_plan(),
        coverage=coverage,negative_controls_rejected=rejected,
        installation="one unfilled condition created; existing completed candidate retained; accepted baseline unchanged",
        compiled_capture="mock identity and reset checked; actual selected compiled exports await user GPU run",
        limitation="CPU AST/address-mask/fixture/probe/file controls only; no actual compilation, hardware bounds check or GPU pass",
        source_sha256={name:t.q.sha(HERE/name) for name in t.NEW_FILES})
    output.parent.mkdir(parents=True,exist_ok=True)
    t.h.write_json(output,report)
    print(f"CPU_FULL_GROUPS_CONTROLS_PASS negative_controls={len(rejected)} GPU_NOT_RUN REPORT={output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B 34-full-groups-cpu-test.py NEW_REPORT_PATH")
    main(Path(sys.argv[1]))
