"""Session 5 (terminal walkthrough): Grouped-Query Attention sweep.

Reference: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints" (2023) — GQA is presented there as an
interpolation between MHA (num_kv_groups=num_heads) and MQA
(num_kv_groups=1) that recovers most of MHA's quality at close to MQA's
cache size, by picking an intermediate number of groups (e.g. 8 for a
32-head model, as in Llama-2-70B / Llama-3).

Reuses `attention.gqa.GroupedQueryAttention` unmodified; sweeps num_kv_groups
and tabulates the resulting cache size via `memory_calc.kv_cache_bytes`.
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from kv_cache_variants.attention.gqa import GroupedQueryAttention
from kv_cache_variants.memory_calc import human_bytes, kv_cache_bytes

console = Console()


def run() -> None:
    console.rule("[bold cyan]Session 5 - Grouped-Query Attention (GQA)")
    torch.manual_seed(0)
    d_model, num_heads, num_layers, seq_len = 256, 8, 12, 4096
    head_dim = d_model // num_heads

    table = Table(title="num_kv_groups sweep (num_heads=8) — GQA interpolates MHA <-> MQA")
    table.add_column("num_kv_groups", justify="right")
    table.add_column("equivalent to", justify="center")
    table.add_column("cached K heads/layer", justify="right")
    table.add_column(f"total cache @ {num_layers} layers, seq_len={seq_len}", justify="right")

    x = torch.randn(1, 5, d_model)
    for num_kv_groups in [1, 2, 4, 8]:
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups)
        _, kv = gqa(x)
        equivalence = "MQA" if num_kv_groups == 1 else ("MHA" if num_kv_groups == num_heads else "-")
        b = kv_cache_bytes(num_layers, num_kv_groups, head_dim, seq_len)
        table.add_row(str(num_kv_groups), equivalence, str(kv[0].shape[1]), human_bytes(b))
    console.print(table)

    console.print(
        "[bold]Note:[/bold] num_kv_groups=1 IS MultiQueryAttention and "
        "num_kv_groups=num_heads IS MultiHeadAttention — verified numerically "
        "in `tests/test_attention_equivalence.py` by copying weights across "
        "modules and asserting identical outputs."
    )


if __name__ == "__main__":
    run()
