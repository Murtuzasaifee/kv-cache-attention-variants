import torch
import torch.nn.functional as F

from kv_cache_variants.sdpa_backends import available_backends, run_sdpa


def test_math_backend_matches_default_sdpa():
    torch.manual_seed(0)
    q = k = v = torch.randn(1, 2, 6, 8)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    actual = run_sdpa(q, k, v, backend="math", is_causal=True)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_unknown_backend_raises():
    q = k = v = torch.randn(1, 2, 4, 8)
    try:
        run_sdpa(q, k, v, backend="nonsense")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_math_always_available():
    assert "math" in available_backends()
