"""Session 9 (terminal walkthrough): migrating hand-rolled attention to SDPA.

Reference: PyTorch `torch.nn.functional.scaled_dot_product_attention` docs —
a single fused, backend-dispatching op replacing hand-rolled
softmax(QK^T/sqrt(d))V. The Session 3 `MultiHeadAttention` module already
calls SDPA internally (see `attention/mha.py`); this walkthrough proves that
migrating the REMAINING hand-rolled math (a manual softmax(QK^T/sqrt(d))V)
to `F.scaled_dot_product_attention` is numerically identical, per the
grading rubric note for Session 9: "a migration that changes outputs is a
bug, not a valid solution."
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

console = Console()


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    """Hand-rolled reference: softmax(QK^T / sqrt(d_k)) V, exactly what SDPA fuses."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if is_causal:
        seq = q.shape[-2]
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def run() -> None:
    console.rule("[bold cyan]Session 9 - Migrating to torch.nn.functional.scaled_dot_product_attention")
    torch.manual_seed(0)

    table = Table(title="manual softmax(QK^T/sqrt(d))V  vs  F.scaled_dot_product_attention")
    table.add_column("shape (batch, heads, seq, head_dim)")
    table.add_column("max abs diff", justify="right")
    table.add_column("numerically identical (atol=1e-5)?", justify="center")
    for shape in [(1, 4, 8, 16), (2, 8, 32, 64), (1, 2, 128, 32)]:
        q, k, v = (torch.randn(shape) for _ in range(3))
        manual_out = manual_attention(q, k, v, is_causal=True)
        sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = (manual_out - sdpa_out).abs().max().item()
        identical = torch.allclose(manual_out, sdpa_out, atol=1e-5)
        table.add_row(str(shape), f"{diff:.2e}", "yes" if identical else "NO - bug")
    console.print(table)

    console.print(
        "[bold]Why migrate anyway if the math is identical:[/bold] SDPA "
        "dispatches to fused, backend-specific kernels (flash/efficient/cudnn, "
        "Session 8) that avoid materializing the full attention matrix and "
        "avoid Python-level overhead — same numbers, less memory, faster."
    )


if __name__ == "__main__":
    run()
