import sys

sys.path.insert(0, ".")
import torch
from starter import fastest_backend


def test_fastest_backend_is_a_valid_name():
    from kv_cache_variants.sdpa_backends import available_backends

    q = k = v = torch.randn(1, 2, 32, 16)
    result = fastest_backend(q, k, v)
    assert result in available_backends()
