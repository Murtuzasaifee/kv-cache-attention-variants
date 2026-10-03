import sys

sys.path.insert(0, ".")
from starter import max_batch_size


def test_max_batch_size_fits_budget():
    n = max_batch_size(budget_bytes=10_000_000, num_layers=4, num_kv_heads=4, head_dim=32, seq_len=100)
    assert n >= 1
    from kv_cache_variants.memory_calc import kv_cache_bytes

    assert kv_cache_bytes(4, 4, 32, 100, batch_size=n) <= 10_000_000
    assert kv_cache_bytes(4, 4, 32, 100, batch_size=n + 1) > 10_000_000
