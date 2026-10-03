"""GQA generalizes MHA and MQA: verify numerical equivalence at matched configs."""

import torch

from kv_cache_variants.attention.gqa import GroupedQueryAttention
from kv_cache_variants.attention.mha import MultiHeadAttention
from kv_cache_variants.attention.mqa import MultiQueryAttention


def _copy_weights(src_proj, dst_proj):
    dst_proj.weight.data.copy_(src_proj.weight.data)


def test_gqa_matches_mha_when_num_kv_groups_equals_num_heads():
    torch.manual_seed(0)
    d_model, num_heads = 16, 4
    mha = MultiHeadAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups=num_heads)

    _copy_weights(mha.q_proj, gqa.q_proj)
    _copy_weights(mha.k_proj, gqa.k_proj)
    _copy_weights(mha.v_proj, gqa.v_proj)
    _copy_weights(mha.out_proj, gqa.out_proj)

    x = torch.randn(2, 5, d_model)
    out_mha, _ = mha(x)
    out_gqa, _ = gqa(x)
    assert torch.allclose(out_mha, out_gqa, atol=1e-5)


def test_gqa_matches_mqa_when_num_kv_groups_is_one():
    torch.manual_seed(0)
    d_model, num_heads = 16, 4
    mqa = MultiQueryAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups=1)

    _copy_weights(mqa._gqa.q_proj, gqa.q_proj)
    _copy_weights(mqa._gqa.k_proj, gqa.k_proj)
    _copy_weights(mqa._gqa.v_proj, gqa.v_proj)
    _copy_weights(mqa._gqa.out_proj, gqa.out_proj)

    x = torch.randn(2, 5, d_model)
    out_mqa, _ = mqa(x)
    out_gqa, _ = gqa(x)
    assert torch.allclose(out_mqa, out_gqa, atol=1e-5)


def test_cache_size_ordering_mha_gqa_mqa():
    """At matched (d_model, num_heads), cache heads shrink MHA > GQA > MQA."""
    torch.manual_seed(0)
    d_model, num_heads = 16, 4
    x = torch.randn(1, 3, d_model)

    _, kv_mha = MultiHeadAttention(d_model, num_heads)(x)
    _, kv_gqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups=2)(x)
    _, kv_mqa = MultiQueryAttention(d_model, num_heads)(x)

    assert kv_mha[0].shape[1] == 4
    assert kv_gqa[0].shape[1] == 2
    assert kv_mqa[0].shape[1] == 1
