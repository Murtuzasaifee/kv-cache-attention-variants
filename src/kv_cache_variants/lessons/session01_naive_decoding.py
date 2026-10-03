"""Session 1 (terminal walkthrough): Naive decoding — why we need a KV cache.

Reference: Vaswani et al., "Attention Is All You Need" (2017) — the scaled
dot-product attention this whole module builds on. Naive decoding applies
that attention correctly but re-invokes it over the FULL sequence on every
generated token, which is the inefficiency Sessions 2-10 exist to fix.

Reuses `naive_decode.generate_naive` unmodified — this file only adds the
rich terminal narration around it, it does not reimplement the decode loop.
"""

from __future__ import annotations

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch import nn

from kv_cache_variants.naive_decode import TinyCausalAttention, generate_naive

console = Console()


def run() -> None:
    console.print(
        Panel(
            "Each new token forces a full forward pass over the ENTIRE sequence "
            "generated so far — attention, and therefore K/V projections, are "
            "recomputed for every past position at every step. Per-step cost "
            "grows with sequence length instead of staying constant.",
            title="Session 1 - Naive Autoregressive Decoding (no KV cache)",
            border_style="cyan",
        )
    )

    torch.manual_seed(0)
    vocab, d_model, num_new = 50, 32, 8
    embed = nn.Embedding(vocab, d_model)
    lm_head = nn.Linear(d_model, vocab, bias=False)
    attn = TinyCausalAttention(d_model)
    prompt = torch.randint(0, vocab, (1, 4))

    # Instrument the forward pass to count how many K/V positions get
    # recomputed at each step, by monkeypatching the module's __call__
    # timing — no change to the underlying algorithm, purely observational.
    recompute_counts: list[int] = []
    original_forward = attn.forward

    def counting_forward(x: torch.Tensor) -> torch.Tensor:
        recompute_counts.append(x.shape[1])  # sequence length recomputed this step
        return original_forward(x)

    attn.forward = counting_forward  # type: ignore[method-assign]

    out = generate_naive(attn, prompt, embed, lm_head, max_new_tokens=num_new)

    table = Table(title="Per-step recompute cost (no cache)")
    table.add_column("decode step", justify="right")
    table.add_column("tokens fed to attention (recomputed)", justify="right")
    table.add_column("would be with a KV cache", justify="right")
    total_naive, total_cached = 0, 0
    for step, n in enumerate(recompute_counts, start=1):
        total_naive += n
        total_cached += 1  # a cache only needs to process the 1 new token
        table.add_row(str(step), str(n), "1 (+ cached K/V reuse)")
    console.print(table)

    console.print(
        f"[bold]Total attention-tokens processed:[/bold] "
        f"naive = {total_naive}, with-cache = {total_cached} "
        f"([green]{total_naive / total_cached:.1f}x[/green] more work without a cache)"
    )
    assert out.shape == (1, prompt.shape[1] + num_new)
    console.print("[bold green]done[/bold green] — see Session 3 (MHA) for the cached version.")


if __name__ == "__main__":
    run()
