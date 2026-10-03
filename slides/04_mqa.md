# Session 4 — Multi-Query Attention (MQA)

- All query heads share ONE K/V head (num_kv_heads = 1)
- Cache shrinks by a factor of num_heads vs MHA at the same head_dim
- Tradeoff: less K/V representational capacity, can hurt quality vs MHA
- Implemented as `GroupedQueryAttention(num_kv_groups=1)` — see Session 5
- Reference: `src/kv_cache_variants/attention/mqa.py`
