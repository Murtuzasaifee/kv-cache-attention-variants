"""Assignment 06 answer key — matched-budget variant comparison."""

import sys

sys.path.insert(0, "../../src")

from kv_cache_variants.memory_calc import kv_cache_bytes


def bytes_per_token(num_heads: int, head_dim: int, num_kv_heads: int, dtype_bytes: int = 2) -> int:
    return kv_cache_bytes(num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim, seq_len=1, dtype_bytes=dtype_bytes)


def mla_bytes_per_token(latent_dim: int, rope_dim: int, dtype_bytes: int = 2) -> int:
    return (latent_dim + rope_dim) * dtype_bytes
