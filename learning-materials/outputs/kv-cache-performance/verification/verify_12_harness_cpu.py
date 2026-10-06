"""Local harness logic checks using NumPy arrays, NOT PyTorch/GPU validation.

The workspace runtime has no torch. A tiny explicitly named stand-in exercises
index selection, corruption controls, and fail-closed behavior only.
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

root = Path(__file__).resolve().parent.parent


class CopyableArray(np.ndarray):
    def clone(self):
        return self.copy()


stand_in = SimpleNamespace(
    float16=np.float16,
    arange=lambda *args, **kwargs: np.arange(*args, **kwargs).view(CopyableArray),
    full=lambda *args, **kwargs: np.full(*args, **kwargs).view(CopyableArray),
    equal=np.array_equal,
    all=np.all,
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def must_raise(exception, operation):
    try:
        operation()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


sys.modules["torch"] = stand_in
harness = load(root / "12-check-kv-indices.py", "harness")
checker = load(root / "12-check-kv-targets.py", "learner_checker").check_kv_targets
for filename, function in (("12-check-kv-indices.py", "main"),
                           ("12-check-kv-targets.py", "check_kv_targets")):
    source = (root / filename).read_text(encoding="utf-8")
    ast.parse(source)
    harness.validate_draft(source, function)
must_raise(RuntimeError, lambda: harness.validate_draft("def check():\n    pass\n", "check"))
must_raise(RuntimeError, lambda: harness.validate_draft("def other():\n    return 1\n", "check"))
controls = harness.checker_controls(stand_in, checker)
assert len(controls["negative_controls"]) == 5
must_raise(RuntimeError, lambda: harness.checker_controls(stand_in, lambda *args: None))


def k_only(kc, vc, ek, ev, rows):
    for pid, pos in enumerate(rows):
        assert np.array_equal(kc[pos], ek[pid])


must_raise(RuntimeError, lambda: harness.checker_controls(stand_in, k_only))
ek = stand_in.arange(256, dtype=np.float16).reshape(2, 128) + 10
ev = ek + 1000
for rows in harness.CASES:
    kc = stand_in.full((8, 128), -1, dtype=np.float16)
    vc = kc.copy()
    idx = np.array(rows, dtype=np.int32)
    for pid, pos in enumerate(rows):
        kc[pos], vc[pos] = ek[pid], ev[pid]
    checker(kc, vc, ek, ev, rows)
    harness.check_other_state(stand_in, kc, vc, ek, ev, idx, ek, ev, idx, rows)
    untouched = next(row for row in range(8) if row not in rows)
    for kind in range(5):
        bad = [array.copy() for array in (kc, vc, ek, ev, idx)]
        if kind < 2:
            bad[kind][untouched, 127] = -99
        else:
            bad[kind].flat[0] = -99
        must_raise(AssertionError, lambda: harness.check_other_state(
            stand_in, *bad, ek, ev, idx, rows
        ))
print(json.dumps({
    "scope": "NumPy stand-in verifies index/check logic only; NOT PyTorch or GPU execution",
    "syntax": "PASS",
    "positive_fixtures": 3,
    "target_corruptions_and_mapping_detected": 5,
    "incomplete_sources_rejected": 2,
    "incomplete_checkers_rejected": 2,
    "untouched_and_input_corruptions_detected": 15,
}, indent=2))
