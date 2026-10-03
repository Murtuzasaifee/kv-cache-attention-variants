# Session 2 — KV Cache Memory Math

- What gets cached: K and V per layer, per head. Q is never cached (only needed for the current step).
- `KV_per_token = 2 x num_kv_heads x head_dim x num_layers x dtype_bytes`
- `KV_cache_bytes = KV_per_token x seq_len x batch_size`
- dtype_bytes: 2 for fp16/bf16 (standard inference), 4 for fp32
- This formula is the SAME formula reused in Sessions 4/5/6 — only `num_kv_heads` (or latent_dim for MLA) changes
- Reference: `src/kv_cache_variants/memory_calc.py::kv_cache_bytes`
