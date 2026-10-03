"""Session 6: Multi-Head Latent Attention (DeepSeek-V2 style).

Fundamentally different from MQA/GQA: instead of caching fewer *heads* of
full-size K/V, MLA caches a single low-rank *latent* vector per token
(dim latent_dim << num_heads*head_dim) and up-projects it back to per-head
K/V at attention time. Cache size ~ (latent_dim + rope_dim), independent of
num_heads.

RoPE doesn't commute with the low-rank down/up projections, so position info
is carried by a small *decoupled* rope key/query pair computed straight from
x (not from the latent) — this is cached alongside the latent, uncompressed
but tiny (rope_dim << head_dim * num_heads).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kv_cache_variants.rope import apply_rotary

MLACache = tuple[Tensor, Tensor]  # (latent, k_rope)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, latent_dim: int, rope_head_dim: int = 16
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_head_dim = rope_head_dim
        self.latent_dim = latent_dim

        self.kv_down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.k_up_proj = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.q_rope_proj = nn.Linear(d_model, num_heads * rope_head_dim, bias=False)
        self.k_rope_proj = nn.Linear(d_model, rope_head_dim, bias=False)  # shared, decoupled

        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def _split(self, x: Tensor, n_heads: int, dim: int) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, n_heads, dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        past_kv: MLACache | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, MLACache]:
        """x: (batch, seq, d_model). rope: optional (cos, sin) from rope.build_rope_cache.

        Returns (output, new_cache) where new_cache = (latent, k_rope) — the
        ONLY things that need to be stored, and both are much smaller than a
        full per-head K/V cache.
        """
        batch, seq, _ = x.shape
        offset = 0 if past_kv is None else past_kv[0].shape[1]

        latent = self.kv_down_proj(x)  # (batch, seq, latent_dim)
        k_rope = self.k_rope_proj(x)  # (batch, seq, rope_head_dim) — shared across heads

        if past_kv is not None:
            past_latent, past_k_rope = past_kv
            latent = torch.cat([past_latent, latent], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)
        new_cache: MLACache = (latent, k_rope)

        k_content = self._split(self.k_up_proj(latent), self.num_heads, self.head_dim)
        v = self._split(self.v_up_proj(latent), self.num_heads, self.head_dim)

        q_content = self._split(self.q_proj(x), self.num_heads, self.head_dim)
        q_rope = self._split(self.q_rope_proj(x), self.num_heads, self.rope_head_dim)
        k_rope_expanded = k_rope.unsqueeze(1)  # (batch, 1, total_seq, rope_head_dim)

        if rope is not None:
            cos, sin = rope
            q_rope = apply_rotary(q_rope, cos, sin, offset=offset)
            k_rope_expanded = apply_rotary(k_rope_expanded, cos, sin, offset=0)
        k_rope_expanded = k_rope_expanded.expand(-1, self.num_heads, -1, -1)

        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope_expanded], dim=-1)

        is_causal = past_kv is None
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.out_proj(out), new_cache


def _demo() -> None:
    torch.manual_seed(0)
    from kv_cache_variants.rope import build_rope_cache

    d_model, num_heads, latent_dim, rope_dim = 32, 4, 8, 8
    mla = MultiHeadLatentAttention(d_model, num_heads, latent_dim, rope_dim)
    cos, sin = build_rope_cache(rope_dim, max_seq_len=16)

    x = torch.randn(1, 5, d_model)
    out, cache = mla(x, rope=(cos, sin))
    assert out.shape == (1, 5, d_model)
    assert cache[0].shape == (1, 5, latent_dim)  # cached latent is tiny vs num_heads*head_dim

    next_x = torch.randn(1, 1, d_model)
    out2, cache2 = mla(next_x, past_kv=cache, rope=(cos, sin))
    assert out2.shape == (1, 1, d_model)
    assert cache2[0].shape == (1, 6, latent_dim)
    print("MLA forward + latent cache OK")


if __name__ == "__main__":
    _demo()
