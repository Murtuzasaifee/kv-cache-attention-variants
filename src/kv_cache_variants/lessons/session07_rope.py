"""Session 7 (terminal walkthrough): Rotary Positional Embeddings (RoPE).

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
Embedding" (RoPE paper, 2021/2023) — encodes absolute position by rotating
Q/K pairs in 2D subspaces, so their dot product depends only on relative
position. Context-extension strategies compared here:
  - Position Interpolation (Chen et al., 2023): compress positions linearly.
  - Dynamic NTK scaling (community/EleutherAI writeup, 2023): stretch the
    rotation base once seq_len exceeds the trained context.
  - YaRN (Peng et al., "YaRN: Efficient Context Window Extension of Large
    Language Models", 2023): per-frequency ramp, leaves local/high-frequency
    dims unscaled to avoid the quality loss pure linear scaling causes.

Reuses `rope.py` unmodified.
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from kv_cache_variants.rope import (
    apply_rotary,
    build_rope_cache,
    dynamic_ntk_rope_cache,
    linear_scaled_rope_cache,
    yarn_rope_cache,
)

console = Console()


def run() -> None:
    console.rule("[bold cyan]Session 7 - RoPE + Context-Extension Scaling")
    torch.manual_seed(0)
    head_dim, trained_len = 16, 8
    extended_len = 32

    cos, sin = build_rope_cache(head_dim, trained_len)
    q = torch.randn(1, 2, trained_len, head_dim)
    rotated = apply_rotary(q, cos, sin)
    norm_preserved = torch.allclose(rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    console.print(f"rotation preserves vector norm (rotation is orthogonal): {norm_preserved}")

    # Incremental decode correctness: rotating one new token at offset=trained_len
    # must match rotating the full extended sequence and slicing the last position.
    cos2, sin2 = build_rope_cache(head_dim, trained_len + 1)
    q_next = torch.randn(1, 2, 1, head_dim)
    full = torch.cat([q, q_next], dim=2)
    rotated_full = apply_rotary(full, cos2, sin2)
    rotated_next = apply_rotary(q_next, cos2, sin2, offset=trained_len)
    matches = torch.allclose(rotated_full[:, :, -1:, :], rotated_next, atol=1e-5)
    console.print(f"incremental single-token rotation matches full recompute: {matches}\n")

    # Compare frequency scaling strategies at the LOWEST frequency (dim 0,
    # slowest-rotating pair) — this is where extension strategies differ most.
    linear_cos, _ = linear_scaled_rope_cache(head_dim, extended_len, scale_factor=extended_len / trained_len)
    ntk_cos, _ = dynamic_ntk_rope_cache(head_dim, extended_len, trained_max_len=trained_len)
    yarn_cos, _ = yarn_rope_cache(
        head_dim, extended_len, trained_max_len=trained_len, scale_factor=extended_len / trained_len
    )
    base_cos, _ = build_rope_cache(head_dim, extended_len)

    table = Table(title=f"Rotation angle (cos) at position {extended_len - 1}, lowest-freq dim, "
                        f"trained_len={trained_len} -> extended_len={extended_len}")
    table.add_column("strategy")
    table.add_column("cos(angle)", justify="right")
    table.add_column("effective position seen by lowest freq", justify="right")
    for name, table_cos in [
        ("no scaling (extrapolates, breaks)", base_cos),
        ("linear / Position Interpolation", linear_cos),
        ("dynamic NTK", ntk_cos),
        ("YaRN (ramped)", yarn_cos),
    ]:
        table.add_row(name, f"{table_cos[-1, 0].item():.4f}", "-")
    console.print(table)
    console.print(
        "[bold]Takeaway:[/bold] unscaled RoPE extrapolates past its trained "
        "range with angles the model never saw during training; all three "
        "scaling strategies keep the effective rotation within (or near) the "
        "trained range, trading off local-attention sharpness differently."
    )


if __name__ == "__main__":
    run()
