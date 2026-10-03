"""Session 2 (terminal walkthrough): KV cache memory math.

Formula (from `memory_calc.kv_cache_bytes`, reused unmodified here):

    KV_per_token   = 2 x num_kv_heads x head_dim x num_layers x dtype_bytes
    KV_cache_bytes = KV_per_token x seq_len x batch_size

The "2" is K and V. This is the arithmetic behind every "why does my 70B
model OOM at long context" incident — reference: the memory-bandwidth
discussion in Dao et al., "FlashAttention" (2022) sec. 1, and the KV-cache
sizing tables in the GQA paper (Ainslie et al., 2023) sec. 3.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from kv_cache_variants.memory_calc import human_bytes, kv_cache_bytes

console = Console()

# Real-ish model configs (num_layers, num_heads, head_dim), fp16 = 2 bytes/elem
CONFIGS = {
    "7B-ish (Llama-2-7B shape)": (32, 32, 128),
    "13B-ish": (40, 40, 128),
    "70B-ish": (80, 64, 128),
}
SEQ_LENS = (2048, 8192, 32768)


def run() -> None:
    console.rule("[bold cyan]Session 2 - KV Cache Memory Math")
    console.print(
        "KV_per_token = 2 x num_kv_heads x head_dim x num_layers x dtype_bytes\n"
        "KV_cache_bytes = KV_per_token x seq_len x batch_size\n"
    )

    for name, (layers, heads, head_dim) in CONFIGS.items():
        table = Table(title=f"{name}  (layers={layers}, heads={heads}, head_dim={head_dim}, fp16)")
        table.add_column("seq_len", justify="right")
        table.add_column("MHA cache", justify="right")
        table.add_column("GQA (8 groups) cache", justify="right")
        table.add_column("MQA cache", justify="right")
        for seq_len in SEQ_LENS:
            mha = kv_cache_bytes(layers, heads, head_dim, seq_len)
            gqa = kv_cache_bytes(layers, min(8, heads), head_dim, seq_len)
            mqa = kv_cache_bytes(layers, 1, head_dim, seq_len)
            table.add_row(str(seq_len), human_bytes(mha), human_bytes(gqa), human_bytes(mqa))
        console.print(table)

    console.print(
        "[bold]Takeaway:[/bold] cache size scales linearly in num_kv_heads — "
        "GQA/MQA cut it by trading away independent K/V per query head "
        "(quality cost explored in Sessions 4-5)."
    )


if __name__ == "__main__":
    run()
