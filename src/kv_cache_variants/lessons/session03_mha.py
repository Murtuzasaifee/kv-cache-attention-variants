"""Session 3 (terminal walkthrough): Multi-Head Attention with incremental KV cache.

Reference: Vaswani et al., "Attention Is All You Need" (2017) for the MHA
math itself; this walkthrough's point is the *caching* pattern layered on
top (K/V per layer stored and concatenated across decode steps), which is
the reference cache shape every later variant (MQA/GQA/MLA) is compared to.

Reuses `attention.mha.MultiHeadAttention` unmodified.
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from kv_cache_variants.attention.mha import MultiHeadAttention
from kv_cache_variants.memory_calc import human_bytes

console = Console()


def run() -> None:
    console.rule("[bold cyan]Session 3 - Multi-Head Attention + Incremental KV Cache")
    torch.manual_seed(0)
    d_model, num_heads = 256, 8
    mha = MultiHeadAttention(d_model, num_heads)

    # Prefill: one forward pass over the prompt, producing the first cache.
    prompt = torch.randn(1, 6, d_model)
    out, kv = mha(prompt)
    console.print(
        f"prefill: prompt_len={prompt.shape[1]}, output shape={tuple(out.shape)}, "
        f"cached K shape={tuple(kv[0].shape)}"
    )

    # Decode: one new token per step, cache grows by exactly one position —
    # this is the payoff versus Session 1's full-recompute.
    table = Table(title="Incremental decode: cache grows by 1 token/step")
    table.add_column("decode step", justify="right")
    table.add_column("cached seq_len", justify="right")
    table.add_column("cache bytes/layer (this layer, fp16)", justify="right")
    dtype_bytes = 2
    for step in range(1, 6):
        next_token = torch.randn(1, 1, d_model)
        out, kv = mha(next_token, past_kv=kv)
        seq_len = kv[0].shape[2]
        bytes_this_layer = 2 * num_heads * (d_model // num_heads) * seq_len * dtype_bytes
        table.add_row(str(step), str(seq_len), human_bytes(bytes_this_layer))
        assert out.shape == (1, 1, d_model)
    console.print(table)
    console.print(
        "[bold green]done[/bold green] — this cache layout (per-layer K/V, "
        "num_heads wide) is the baseline Sessions 4-6 compress."
    )


if __name__ == "__main__":
    run()
