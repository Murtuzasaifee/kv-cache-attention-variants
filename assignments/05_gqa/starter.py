"""Assignment 05 answer key — pick num_kv_groups for a cache budget."""

import sys

sys.path.insert(0, "../../src")

from kv_cache_variants.memory_calc import kv_cache_bytes


def largest_groups_within_budget(
    budget_bytes: int, num_heads: int, num_layers: int, head_dim: int, seq_len: int
) -> int:
    for groups in range(num_heads, 0, -1):
        if num_heads % groups != 0:
            continue
        if kv_cache_bytes(num_layers, groups, head_dim, seq_len) <= budget_bytes:
            return groups
    return 1
