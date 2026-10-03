from kv_cache_variants.memory_calc import human_bytes, kv_cache_bytes


def test_kv_cache_bytes_formula():
    assert kv_cache_bytes(num_layers=2, num_kv_heads=4, head_dim=8, seq_len=10) == 2 * 4 * 8 * 2 * 2 * 10


def test_kv_cache_bytes_scales_with_batch():
    one = kv_cache_bytes(num_layers=2, num_kv_heads=4, head_dim=8, seq_len=10, batch_size=1)
    two = kv_cache_bytes(num_layers=2, num_kv_heads=4, head_dim=8, seq_len=10, batch_size=2)
    assert two == one * 2


def test_human_bytes():
    assert human_bytes(0) == "0.00 B"
    assert human_bytes(1024) == "1.00 KB"
    assert human_bytes(1024 * 1024) == "1.00 MB"
