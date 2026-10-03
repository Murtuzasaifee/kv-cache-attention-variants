"""Assignment 03 answer key — MHA incremental decode equivalence."""

import sys

sys.path.insert(0, "../../src")

import torch
from kv_cache_variants.attention.mha import MultiHeadAttention


def decode_incrementally(mha: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    outputs = []
    cache = None
    for t in range(x.shape[1]):
        out, cache = mha(x[:, t : t + 1, :], past_kv=cache)
        outputs.append(out)
    return torch.cat(outputs, dim=1)
