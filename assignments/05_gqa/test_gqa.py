import sys

sys.path.insert(0, ".")
from starter import largest_groups_within_budget


def test_returns_valid_divisor():
    groups = largest_groups_within_budget(
        budget_bytes=1_500_000, num_heads=8, num_layers=4, head_dim=64, seq_len=1000
    )
    assert 8 % groups == 0
    from kv_cache_variants.memory_calc import kv_cache_bytes

    assert kv_cache_bytes(4, groups, 64, 1000) <= 1_500_000
