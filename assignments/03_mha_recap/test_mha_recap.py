import sys

sys.path.insert(0, ".")
import torch
from starter import decode_incrementally
from kv_cache_variants.attention.mha import MultiHeadAttention


def test_incremental_matches_full_forward():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(1, 5, 16)
    full_out, _ = mha(x)
    incremental_out = decode_incrementally(mha, x)
    assert torch.allclose(full_out, incremental_out, atol=1e-5)
