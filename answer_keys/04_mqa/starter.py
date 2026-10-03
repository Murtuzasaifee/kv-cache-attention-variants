"""Assignment 04 answer key — MQA cache savings."""

import sys

sys.path.insert(0, "../../src")

from kv_cache_variants.memory_calc import kv_cache_bytes


def mqa_savings_factor(num_heads: int, num_layers: int, head_dim: int, seq_len: int) -> float:
    mha_bytes = kv_cache_bytes(num_layers, num_heads, head_dim, seq_len)
    mqa_bytes = kv_cache_bytes(num_layers, 1, head_dim, seq_len)
    return mha_bytes / mqa_bytes
