# Session 3 — Multi-Head Attention Recap

- Q, K, V each projected to `num_heads` heads of `head_dim = d_model / num_heads`
- Attention per head: `softmax(QK^T / sqrt(head_dim)) @ V`, heads concatenated then output-projected
- KV cache shape: `(batch, num_heads, seq, head_dim)` per layer — this is the reference shape for every later variant
- Standalone baseline (this repo builds it from scratch, no external dependency)
- Reference: `src/kv_cache_variants/attention/mha.py`
