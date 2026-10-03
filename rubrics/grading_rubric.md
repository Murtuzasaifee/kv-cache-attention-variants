# Grading Rubric — KV Cache & Attention Variants (Module 03)

Applies to all 10 assignments. Total: 10 points per assignment unless noted.

| Criterion | Points | Notes |
|---|---|---|
| Correctness | 5 | `test_*.py` in the assignment directory passes unmodified |
| Numerical/conceptual justification | 2 | Student can explain WHY the result holds (e.g. why GQA cache scales with num_kv_groups, why RoPE offset must equal past_len), not just that the test passes |
| Code quality | 2 | Type-annotated, no dead code, reuses `memory_calc`/`bench`/`rope` helpers rather than reimplementing them |
| Original example | 1 | Uses a model config or scenario NOT copied verbatim from Raschka/DeepSeek-V2/GQA-paper diagrams |

## Session-Specific Notes

- **Sessions 4-6 (MQA/GQA/MLA):** must reuse `memory_calc.kv_cache_bytes` — reimplementing the formula from scratch loses code-quality points.
- **Session 6 comparison:** grade at a MATCHED cache budget across variants (bytes/token), not matched head count — an MLA with latent_dim=head_dim isn't a fair comparison to MHA.
- **Session 8:** local benchmark results are backend-*availability* dependent (device-specific) — grade the methodology (using `bench.benchmark` correctly, interpreting results), not a fixed expected number.
- **Session 9:** confirm the student's SDPA migration is numerically identical to the hand-rolled Session 3 module (see `test_migration_equivalence.py` pattern) — a "migration" that changes outputs is a bug, not a valid solution.
- **Session 10:** graded on the Colab notebook run (screenshot/output log of vLLM throughput vs naive `generate()`), since it requires GPU access unavailable locally.

## Pass Threshold

7/10 per assignment to pass the module. All 10 assignments must be attempted.
