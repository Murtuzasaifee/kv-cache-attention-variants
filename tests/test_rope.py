import torch

from kv_cache_variants.rope import apply_rotary, build_rope_cache, yarn_rope_cache


def test_rotation_preserves_norm():
    head_dim, seq_len = 16, 8
    cos, sin = build_rope_cache(head_dim, seq_len)
    q = torch.randn(1, 2, seq_len, head_dim)
    rotated = apply_rotary(q, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_offset_rotation_matches_full_sequence_slice():
    head_dim, seq_len = 16, 8
    cos, sin = build_rope_cache(head_dim, seq_len + 1)
    q = torch.randn(1, 2, seq_len, head_dim)
    q_next = torch.randn(1, 2, 1, head_dim)
    full = torch.cat([q, q_next], dim=2)

    rotated_full = apply_rotary(full, cos, sin)
    rotated_next = apply_rotary(q_next, cos, sin, offset=seq_len)
    assert torch.allclose(rotated_full[:, :, -1:, :], rotated_next, atol=1e-5)


def test_yarn_cache_shape():
    cos, sin = yarn_rope_cache(head_dim=16, seq_len=8, trained_max_len=4, scale_factor=2.0)
    assert cos.shape == (8, 16)
    assert sin.shape == (8, 16)
