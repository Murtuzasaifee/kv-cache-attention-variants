# KV Cache Memory Worksheet

Formula (see `src/kv_cache_variants/memory_calc.py::kv_cache_bytes`):

```
KV_per_token   = 2 x num_kv_heads x head_dim x num_layers x dtype_bytes
KV_cache_bytes = KV_per_token x seq_len x batch_size
```

`num_kv_heads` per variant, at matched `num_heads`:
- MHA: `num_kv_heads = num_heads`
- GQA: `num_kv_heads = num_kv_groups` (1 < num_kv_groups < num_heads)
- MQA: `num_kv_heads = 1`
- MLA: not head-based — cached bytes/token = `(latent_dim + rope_dim) x dtype_bytes` (see `attention/mla.py`)

## Worked Examples (original configs, fp16 = 2 bytes/elem)

| Config | Layers | Heads | Head dim | Variant | seq_len | Cache size |
|---|---|---|---|---|---|---|
| A | 32 | 32 | 128 | MHA | 4096 | 2.00 GB |
| A | 32 | 32 | 128 | GQA (8 groups) | 4096 | 512 MB |
| A | 32 | 32 | 128 | MQA | 4096 | 64 MB |
| B | 24 | 16 | 96  | MHA | 8192 | 1.12 GB |
| B | 24 | 16 | 96  | GQA (4 groups) | 8192 | 288 MB |
| C | 12 | 8  | 64  | MLA (latent=64, rope=16) | 2048 | 3.75 MB |

Fill in your own rows using `uv run python -c "from kv_cache_variants.memory_calc import kv_cache_bytes, human_bytes; print(human_bytes(kv_cache_bytes(...)))"`.

## Exercise

For a target deployment budget (e.g. 16 GB KV cache on a single GPU), work
backward: given `num_layers`, `head_dim`, `seq_len`, and `batch_size`, what's
the largest `num_kv_heads` (MHA/GQA/MQA) or `latent_dim` (MLA) that fits?
See Assignments 02, 04, 05, 06.
