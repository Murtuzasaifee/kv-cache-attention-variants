# Session 6 — Multi-Head Latent Attention (MLA, DeepSeek-V2 style)

- Fundamentally different from MQA/GQA: compresses WHAT's cached (a low-rank latent), not the number of heads
- Cache per token ~ latent_dim + rope_dim, independent of num_heads
- RoPE doesn't commute with the low-rank down/up-projection, so a small DECOUPLED rope key/query pair is cached uncompressed alongside the latent
- Comparison assignment: MHA vs MQA vs GQA vs MLA at matched cache budget (bytes/token) — which preserves the most quality per cached byte?
- Reference: `src/kv_cache_variants/attention/mla.py`
