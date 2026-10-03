"""Session 3: from-scratch Multi-Head Attention with an incremental KV cache.

Standalone pedagogical baseline (no external "Transformers 101" dependency).
Every later variant (MQA, GQA, MLA) is compared against this at a matched
cache budget, so its cache layout — a list of per-layer (K, V) tensors shaped
(batch, num_heads, seq, head_dim) — is the reference shape for the rest of
the module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

KVCache = list[tuple[Tensor, Tensor]]


class MultiHeadAttention(nn.Module):
    """Classic MHA: one K/V head per query head, cached across decode steps."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: Tensor, past_kv: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """x: (batch, seq, d_model). past_kv, if given, is prepended before attending.

        Returns (output, new_kv) where new_kv is the full (K, V) to cache and
        pass back in on the next decode step.
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        is_causal = past_kv is None  # single growing prompt: causal only on first pass
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(out), (k, v)


def _demo() -> None:
    torch.manual_seed(0)
    d_model, num_heads = 32, 4
    mha = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(1, 5, d_model)
    out, kv = mha(x)
    assert out.shape == (1, 5, d_model)
    assert kv[0].shape == (1, num_heads, 5, d_model // num_heads)

    # incremental decode: one new token, reusing cached kv
    next_x = torch.randn(1, 1, d_model)
    out2, kv2 = mha(next_x, past_kv=kv)
    assert out2.shape == (1, 1, d_model)
    assert kv2[0].shape == (1, num_heads, 6, d_model // num_heads)
    print("MHA forward + incremental cache OK")


if __name__ == "__main__":
    _demo()
