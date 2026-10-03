"""Session 4: Multi-Query Attention — all query heads share a single K/V head.

MQA is the extreme case of GQA with num_kv_groups=1: maximum cache
compression (see memory_calc.py), at the cost of representational capacity
in the shared K/V head.
"""

from __future__ import annotations

from torch import Tensor, nn

from kv_cache_variants.attention.gqa import GroupedQueryAttention


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self._gqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups=1)

    def forward(
        self, x: Tensor, past_kv: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return self._gqa(x, past_kv)


def _demo() -> None:
    import torch

    torch.manual_seed(0)
    d_model, num_heads = 32, 4
    mqa = MultiQueryAttention(d_model, num_heads)
    x = torch.randn(1, 5, d_model)
    out, kv = mqa(x)
    assert out.shape == (1, 5, d_model)
    assert kv[0].shape == (1, 1, 5, d_model // num_heads)  # single shared KV head
    print("MQA forward + single-head cache OK")


if __name__ == "__main__":
    _demo()
