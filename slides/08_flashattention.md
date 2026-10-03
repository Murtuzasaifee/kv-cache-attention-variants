# Session 8 — FlashAttention

- Core insight: the bottleneck is memory bandwidth (HBM <-> SRAM traffic), not FLOPs
- Tiling: compute attention in blocks that fit in fast on-chip SRAM, avoid materializing the full seq x seq attention matrix in HBM
- Kernel fusion: softmax + matmuls fused into one kernel pass
- Result: 2-8x speedup, and importantly, LESS memory, not just faster
- Mac-first note: SDPA's backend dispatch mechanism is the teaching vehicle locally (`sdpa_kernel`); the true fused CUDA kernel needs a GPU (see `colab/session08_flashattention_colab.ipynb`)
- Reference: `src/kv_cache_variants/sdpa_backends.py`, `src/kv_cache_variants/bench.py`
