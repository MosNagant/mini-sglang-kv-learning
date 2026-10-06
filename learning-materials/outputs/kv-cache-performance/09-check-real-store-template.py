"""Your first check of a result produced by the real CUDA store kernel.

The coach's runner prepares the inputs and calls the original kernel.
You write the check below. Row numbers start at 0.
"""

import torch


def check_first_k(k_cache, k):
    # k has 2 rows; k_cache has 8 rows. Each row has 128 numbers.
    # The CUDA call used indices = [5, 1].
    # Task: check the destination of input K row 0, comparing the WHOLE row.
    # Replace the next line with your own assert statement.
    raise NotImplementedError("TODO: write the first K row assertion")
