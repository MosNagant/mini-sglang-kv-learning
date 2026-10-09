"""Learner's reviewed 2026-10-02 submission, formatted as a callable check.

The coach adds only this function wrapper/import. The caller supplies CPU
snapshots of results and independent pre-launch expected inputs.
"""

import torch


def check_kv_targets(k_cache, v_cache, expected_k, expected_v, target_rows):
    for pid, pos in enumerate(target_rows):
        assert torch.equal(k_cache[pos], expected_k[pid])
        assert torch.equal(v_cache[pos], expected_v[pid])
