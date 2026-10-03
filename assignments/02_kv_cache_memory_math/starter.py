"""Assignment 02 answer key — memory worksheet."""

import sys

sys.path.insert(0, "../../src")

from kv_cache_variants.memory_calc import kv_cache_bytes


def max_batch_size(budget_bytes: int, num_layers: int, num_kv_heads: int, head_dim: int, seq_len: int) -> int:
    per_seq = kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch_size=1)
    return budget_bytes // per_seq
