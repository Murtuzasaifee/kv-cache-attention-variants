"""Session 1: naive autoregressive decoding — no KV cache.

Baseline used to motivate caching: each new token forces a full forward pass
over the entire sequence so far (recomputing K/V for every past position),
making per-step cost grow with sequence length instead of being O(1).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class TinyCausalAttention(nn.Module):
    """Minimal single-head causal self-attention block for teaching demos."""

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)."""
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
        ).squeeze(1)
        return self.out(attn)


@torch.no_grad()
def generate_naive(
    model: Callable[[Tensor], Tensor],
    prompt_ids: Tensor,
    embed: nn.Embedding,
    lm_head: nn.Linear,
    max_new_tokens: int,
) -> Tensor:
    """Greedy decode by recomputing the full forward pass every step (no cache).

    Args:
        model: callable mapping embeddings (batch, seq, d_model) -> same shape.
        prompt_ids: (batch, seq) starting token ids.
        embed: token embedding table.
        lm_head: projects model output to vocab logits.
        max_new_tokens: number of tokens to generate.

    Returns:
        (batch, seq + max_new_tokens) token ids.
    """
    ids = prompt_ids
    for _ in range(max_new_tokens):
        x = embed(ids)  # recompute embeddings for the WHOLE sequence, every step
        hidden = model(x)  # recompute attention over the WHOLE sequence, every step
        next_id = lm_head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def _demo() -> None:
    torch.manual_seed(0)
    vocab, d_model = 50, 32
    embed = nn.Embedding(vocab, d_model)
    lm_head = nn.Linear(d_model, vocab, bias=False)
    attn = TinyCausalAttention(d_model)

    prompt = torch.randint(0, vocab, (1, 4))
    out = generate_naive(attn, prompt, embed, lm_head, max_new_tokens=8)
    assert out.shape == (1, 12)
    print(f"naive decode output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    _demo()
