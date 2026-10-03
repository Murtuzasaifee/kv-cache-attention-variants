"""Session 4 (terminal walkthrough): Multi-Query Attention vs MHA.

Reference: Shazeer, "Fast Transformer Decoding: One Write-Head is All You
Need" (2019) — the original MQA paper. All query heads share ONE K/V head,
so the cache no longer scales with num_heads at all.

Reuses `attention.mqa.MultiQueryAttention` and `attention.mha.MultiHeadAttention`
unmodified; this file only benchmarks/tabulates their cache sizes side by side.
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from kv_cache_variants.attention.mha import MultiHeadAttention
from kv_cache_variants.attention.mqa import MultiQueryAttention
from kv_cache_variants.memory_calc import human_bytes, kv_cache_bytes

console = Console()


def run() -> None:
    console.rule("[bold cyan]Session 4 - Multi-Query Attention (MQA)")
    torch.manual_seed(0)
    d_model, num_heads, num_layers, seq_len = 256, 8, 12, 4096
    head_dim = d_model // num_heads

    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)

    x = torch.randn(1, 5, d_model)
    _, kv_mha = mha(x)
    _, kv_mqa = mqa(x)
    console.print(f"MHA cached K heads per layer: {kv_mha[0].shape[1]}")
    console.print(f"MQA cached K heads per layer: {kv_mqa[0].shape[1]}  <- shared by all {num_heads} query heads")

    table = Table(title=f"Full-model cache @ seq_len={seq_len}, {num_layers} layers, fp16")
    table.add_column("variant")
    table.add_column("num_kv_heads", justify="right")
    table.add_column("total KV cache", justify="right")
    for name, n_kv in [("MHA", num_heads), ("MQA", 1)]:
        b = kv_cache_bytes(num_layers, n_kv, head_dim, seq_len)
        table.add_row(name, str(n_kv), human_bytes(b))
    console.print(table)
    console.print(
        f"[bold]Compression ratio:[/bold] {num_heads}x smaller cache, at the cost of "
        "every query head reading from the same K/V representation "
        "(quality/capacity tradeoff — see assignment for empirical probe)."
    )


if __name__ == "__main__":
    run()
