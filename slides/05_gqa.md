# Session 5 — Grouped-Query Attention (GQA)

- Generalizes MHA and MQA: query heads split into `num_kv_groups` groups, heads within a group share one K/V head
- `num_kv_groups = num_heads` -> MHA. `num_kv_groups = 1` -> MQA.
- Lets you dial in a cache-budget/quality tradeoff between the two extremes
- Session 6 comparison assignment uses this to compare MHA/MQA/GQA/MLA at MATCHED cache budgets — apples to apples
- Reference: `src/kv_cache_variants/attention/gqa.py`
