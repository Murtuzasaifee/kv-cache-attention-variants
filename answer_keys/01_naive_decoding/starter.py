"""Assignment 01 answer key — naive decoding."""

import sys

sys.path.insert(0, "../../src")

from kv_cache_variants.naive_decode import TinyCausalAttention, generate_naive  # noqa: F401


def count_flops_estimate(seq_len: int, d_model: int) -> int:
    """Rough FLOPs estimate for one full self-attention forward pass over seq_len tokens."""
    return 2 * seq_len * seq_len * d_model
