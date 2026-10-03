import sys

sys.path.insert(0, ".")
import torch
from starter import raw_sdpa_forward
from kv_cache_variants.attention.mha import MultiHeadAttention


def test_raw_sdpa_matches_module_forward():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(1, 5, 16)
    expected, _ = mha(x)
    actual = raw_sdpa_forward(mha, x)
    assert torch.allclose(expected, actual, atol=1e-5)
