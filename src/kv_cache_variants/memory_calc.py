"""Session 2: KV cache memory math.

KV_per_token = 2 (K and V) x num_kv_heads x head_dim x num_layers x dtype_bytes
KV_cache_bytes = KV_per_token x seq_len x batch_size
"""

from __future__ import annotations


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Total KV cache size in bytes.

    Args:
        num_layers: transformer layer count.
        num_kv_heads: number of KV heads actually cached (== num_query_heads for
            MHA, 1 for MQA, num_groups for GQA). For MLA use the compressed
            latent dim as head_dim with num_kv_heads=1 instead.
        head_dim: per-head dimension.
        seq_len: context length (tokens cached).
        batch_size: concurrent sequences.
        dtype_bytes: bytes per cached element (2 for fp16/bf16, 4 for fp32).

    Returns:
        Total bytes required to store K and V caches across all layers.
    """
    per_token = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes
    return per_token * seq_len * batch_size


def human_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (KB/MB/GB)."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def _demo() -> None:
    # Llama-2-7B-ish config: 32 layers, 32 heads, head_dim 128, fp16.
    b = kv_cache_bytes(num_layers=32, num_kv_heads=32, head_dim=128, seq_len=4096)
    assert b == 2 * 32 * 128 * 32 * 2 * 4096
    assert human_bytes(1024) == "1.00 KB"
    print(f"MHA 7B @ 4096 ctx: {human_bytes(b)}")


if __name__ == "__main__":
    _demo()
