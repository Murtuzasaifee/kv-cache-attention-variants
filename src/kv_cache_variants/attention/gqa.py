"""Session 5: Grouped-Query Attention — generalizes MHA and MQA.

num_kv_groups == num_heads  -> identical to MultiHeadAttention (mha.py)
num_kv_groups == 1          -> identical to MultiQueryAttention (mqa.py)

Query heads are split into num_kv_groups groups; every query head in a group
shares one K/V head. Only num_kv_groups K/V heads are cached, so cache size
scales with num_kv_groups instead of num_heads (see memory_calc.py).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_groups: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must divide evenly by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_model // num_heads
        self.group_size = num_heads // num_kv_groups

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def _split(self, x: Tensor, n_heads: int) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: Tensor, past_kv: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """x: (batch, seq, d_model) -> (output, new_kv). new_kv has num_kv_groups heads (the cached size)."""
        q = self._split(self.q_proj(x), self.num_heads)
        k = self._split(self.k_proj(x), self.num_kv_groups)
        v = self._split(self.v_proj(x), self.num_kv_groups)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        cached_kv = (k, v)  # this is what gets stored — num_kv_groups heads only
        k = k.repeat_interleave(self.group_size, dim=1)  # expand for attention math
        v = v.repeat_interleave(self.group_size, dim=1)

        is_causal = past_kv is None
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(out), cached_kv


def _demo() -> None:
    torch.manual_seed(0)
    d_model, num_heads, num_kv_groups = 32, 4, 2
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups)

    x = torch.randn(1, 5, d_model)
    out, kv = gqa(x)
    assert out.shape == (1, 5, d_model)
    assert kv[0].shape == (1, num_kv_groups, 5, d_model // num_heads)  # smaller cache than MHA
    print("GQA forward + grouped cache OK")


if __name__ == "__main__":
    _demo()
