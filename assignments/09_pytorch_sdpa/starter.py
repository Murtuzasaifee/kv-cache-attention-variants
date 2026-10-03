"""Assignment 09 answer key — migration equivalence."""

import sys

sys.path.insert(0, "../../src")

import torch
import torch.nn.functional as F
from kv_cache_variants.attention.mha import MultiHeadAttention


def raw_sdpa_forward(mha: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    batch, seq, _ = x.shape
    q = mha._split_heads(mha.q_proj(x))
    k = mha._split_heads(mha.k_proj(x))
    v = mha._split_heads(mha.v_proj(x))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).reshape(batch, seq, -1)
    return mha.out_proj(out)
