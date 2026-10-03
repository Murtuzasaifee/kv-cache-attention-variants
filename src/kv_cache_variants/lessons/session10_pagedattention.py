"""Session 10 (terminal walkthrough): PagedAttention's memory-allocation idea.

Reference: Kwon et al., "Efficient Memory Management for Large Language
Model Serving with PagedAttention" (vLLM paper, 2023). Core idea: instead of
preallocating one contiguous KV-cache buffer per sequence sized for its
WORST-CASE max length (like the naive allocator simulated below), split the
cache into fixed-size blocks and allocate them on demand via a per-sequence
block table — like OS virtual memory paging. This eliminates the internal
fragmentation that comes from over-provisioning for max_len.

This is a from-scratch, CPU-only ALLOCATION-STRATEGY simulation to teach the
paging concept with real numbers — not a reimplementation of vLLM's actual
CUDA paged-attention kernel. The real vLLM install + throughput benchmark
(vs naive HF `generate()`) is in
`colab/session10_vllm_pagedattention_colab.ipynb` (GPU required), per the
grading rubric note for Session 10.
"""

from __future__ import annotations

import random

from rich.console import Console
from rich.table import Table

from kv_cache_variants.memory_calc import human_bytes, kv_cache_bytes

console = Console()


def run() -> None:
    console.rule("[bold cyan]Session 10 - PagedAttention: block-table allocation")
    random.seed(0)
    num_layers, num_kv_heads, head_dim, max_len, block_size = 24, 8, 128, 2048, 16

    # A batch of concurrent requests with varying ACTUAL generated lengths
    # (this variance is exactly what makes naive worst-case preallocation wasteful).
    actual_lengths = [random.randint(50, max_len) for _ in range(32)]

    per_token_bytes = 2 * num_kv_heads * head_dim * num_layers * 2  # dtype_bytes=2 (fp16)

    naive_total = sum(kv_cache_bytes(num_layers, num_kv_heads, head_dim, max_len) for _ in actual_lengths)
    # Paged: only allocate the blocks actually needed, rounded up to block_size.
    paged_total = sum(
        kv_cache_bytes(
            num_layers, num_kv_heads, head_dim,
            seq_len=-(-length // block_size) * block_size,  # ceil to block_size
        )
        for length in actual_lengths
    )
    actually_used_total = sum(kv_cache_bytes(num_layers, num_kv_heads, head_dim, length) for length in actual_lengths)

    table = Table(title=f"{len(actual_lengths)} concurrent requests, max_len={max_len}, block_size={block_size}")
    table.add_column("allocation strategy")
    table.add_column("total bytes allocated", justify="right")
    table.add_column("wasted vs. actually used", justify="right")
    table.add_row("naive (preallocate max_len per request)", human_bytes(naive_total),
                  human_bytes(naive_total - actually_used_total))
    table.add_row("paged (block_size-rounded, per-request)", human_bytes(paged_total),
                  human_bytes(paged_total - actually_used_total))
    table.add_row("actually used (lower bound)", human_bytes(actually_used_total), "-")
    console.print(table)

    console.print(
        f"[bold]Memory saved by paging:[/bold] "
        f"{(1 - paged_total / naive_total) * 100:.1f}% less allocated than naive worst-case "
        f"preallocation, bounded to at most block_size-1 tokens of waste per "
        f"request instead of (max_len - actual_length).\n\n"
        f"per-token cache cost this config: {human_bytes(per_token_bytes)}/token — "
        "this is what gets freed/reused per block as sequences finish, letting "
        "vLLM pack more concurrent requests into the same GPU memory."
    )


if __name__ == "__main__":
    run()
