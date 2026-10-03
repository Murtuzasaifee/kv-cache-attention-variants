# Session 9 — PyTorch scaled_dot_product_attention

- Bridge point: hand-rolled attention modules (Sessions 3-6) get replaced by `F.scaled_dot_product_attention`
- Backend dispatch handled by PyTorch: math (always), memory-efficient, FlashAttention-2, cuDNN — picks automatically, or force with `sdpa_kernel`
- `torch.compile`-friendly, unlike hand-rolled attention with Python-level control flow
- Reuses the Session 8 benchmark harness UNCHANGED — same `bench.benchmark` call
- Reference: `src/kv_cache_variants/sdpa_backends.py`
