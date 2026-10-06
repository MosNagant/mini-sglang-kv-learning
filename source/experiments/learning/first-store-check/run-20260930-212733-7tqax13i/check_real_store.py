"""Your first check of a result produced by the real CUDA store kernel.

The coach's runner prepares the inputs and calls the original kernel.
You write the check below. Row numbers start at 0.
"""

import torch


def check_first_k(k_cache, k):

    assert torch.equal(k_cache[5], k[0])