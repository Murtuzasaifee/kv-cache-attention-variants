"""Session 7: Rotary Positional Embeddings (RoPE) + context-extension scaling.

RoPE is applied to Q/K *before* they're cached, so cached K already carries
position info baked in — composes directly with mha/mqa/gqa/mla via
`apply_rotary(..., offset=past_len)` at each decode step.
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_rope_cache(
    head_dim: int, max_seq_len: int, base: float = 10000.0, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables, shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor:
    """x: (batch, heads, seq, head_dim). cos/sin: (max_seq_len, head_dim) from build_rope_cache.

    offset lets you rotate a single new decode-step token by its true position
    (past_len) rather than always starting from 0.
    """
    seq = x.shape[2]
    c = cos[offset : offset + seq].to(x.dtype)
    s = sin[offset : offset + seq].to(x.dtype)
    return x * c + _rotate_half(x) * s


def linear_scaled_rope_cache(
    head_dim: int, max_seq_len: int, scale_factor: float, base: float = 10000.0
) -> tuple[Tensor, Tensor]:
    """Position Interpolation: compress positions by scale_factor so a model
    trained on a shorter context can attend over max_seq_len * scale_factor."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float() / scale_factor
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def dynamic_ntk_rope_cache(
    head_dim: int, seq_len: int, trained_max_len: int, base: float = 10000.0
) -> tuple[Tensor, Tensor]:
    """Dynamic NTK scaling: stretch the frequency base itself once seq_len
    exceeds the trained context, instead of compressing positions."""
    if seq_len > trained_max_len:
        base = base * ((seq_len / trained_max_len) - (seq_len / trained_max_len - 1))
        base = base * (seq_len / trained_max_len) ** (head_dim / (head_dim - 2))
    return build_rope_cache(head_dim, seq_len, base=base)


def yarn_rope_cache(
    head_dim: int,
    seq_len: int,
    trained_max_len: int,
    scale_factor: float,
    base: float = 10000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Simplified YaRN: low frequencies (long-range) get linear-interpolated,
    high frequencies (local/short-range) are left untouched, with a smooth
    ramp between the two — avoids YaRN's known degradation of local attention
    that pure linear scaling causes.
    """
    dim_range = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / (base ** (dim_range / head_dim))

    # ramp: 0 = keep high-freq as-is, 1 = fully linear-interpolate
    low = head_dim * torch.log(torch.tensor(trained_max_len / (beta_fast * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(base))
    )
    high = head_dim * torch.log(torch.tensor(trained_max_len / (beta_slow * 2 * torch.pi))) / (
        2 * torch.log(torch.tensor(base))
    )
    ramp = ((dim_range / 2 - low) / (high - low + 1e-3)).clamp(0, 1)

    inv_freq_interp = inv_freq / scale_factor
    inv_freq_final = inv_freq * (1 - ramp) + inv_freq_interp * ramp

    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq_final)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _demo() -> None:
    torch.manual_seed(0)
    head_dim, seq_len = 16, 8
    cos, sin = build_rope_cache(head_dim, seq_len)
    assert cos.shape == (seq_len, head_dim)

    q = torch.randn(1, 2, seq_len, head_dim)
    rotated = apply_rotary(q, cos, sin)
    assert rotated.shape == q.shape
    # rotation preserves vector norm per position
    assert torch.allclose(rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-4)

    # incremental: rotating one new token at offset=seq_len matches rotating
    # the full extended sequence and slicing the last position
    cos2, sin2 = build_rope_cache(head_dim, seq_len + 1)
    q_next = torch.randn(1, 2, 1, head_dim)
    full = torch.cat([q, q_next], dim=2)
    rotated_full = apply_rotary(full, cos2, sin2)
    rotated_next = apply_rotary(q_next, cos2, sin2, offset=seq_len)
    assert torch.allclose(rotated_full[:, :, -1:, :], rotated_next, atol=1e-5)

    linear_scaled_rope_cache(head_dim, seq_len, scale_factor=2.0)
    dynamic_ntk_rope_cache(head_dim, seq_len, trained_max_len=4)
    yarn_rope_cache(head_dim, seq_len, trained_max_len=4, scale_factor=2.0)
    print("RoPE base + linear/dynamic-ntk/yarn scaling OK")


if __name__ == "__main__":
    _demo()
