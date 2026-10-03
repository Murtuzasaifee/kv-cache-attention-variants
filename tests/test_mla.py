import torch

from kv_cache_variants.attention.mla import MultiHeadLatentAttention
from kv_cache_variants.rope import build_rope_cache


def test_mla_incremental_cache_matches_full_forward():
    """Decoding one token at a time with cache should match a single full-sequence pass."""
    torch.manual_seed(0)
    d_model, num_heads, latent_dim, rope_dim = 16, 2, 6, 8
    mla = MultiHeadLatentAttention(d_model, num_heads, latent_dim, rope_dim)
    cos, sin = build_rope_cache(rope_dim, max_seq_len=10)

    x = torch.randn(1, 4, d_model)
    full_out, _ = mla(x, rope=(cos, sin))

    # incremental: feed one token at a time
    out_incremental = []
    cache = None
    for t in range(x.shape[1]):
        step_out, cache = mla(x[:, t : t + 1, :], past_kv=cache, rope=(cos, sin))
        out_incremental.append(step_out)
    incremental_out = torch.cat(out_incremental, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-5)


def test_mla_cache_smaller_than_full_kv():
    d_model, num_heads, latent_dim, rope_dim = 32, 4, 8, 8
    head_dim = d_model // num_heads
    mla = MultiHeadLatentAttention(d_model, num_heads, latent_dim, rope_dim)
    x = torch.randn(1, 5, d_model)
    cos, sin = build_rope_cache(rope_dim, max_seq_len=8)
    _, (latent, k_rope) = mla(x, rope=(cos, sin))

    cached_elems_per_token = latent.shape[-1] + k_rope.shape[-1]
    full_kv_elems_per_token = 2 * num_heads * head_dim
    assert cached_elems_per_token < full_kv_elems_per_token
