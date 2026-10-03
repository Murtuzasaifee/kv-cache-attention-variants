# Assignment 01 — Naive Decoding

## Objective
Implement `generate_naive_fixed_steps` that greedily decodes `n` tokens by
recomputing the full forward pass every step (no cache), and measure how
per-step wall-clock time grows with sequence length.

## Starter
See `starter.py`. Fill in `TODO` in `count_flops_estimate`.

## Acceptance Criteria
- `test_naive_decoding.py` passes.
- You can explain, in one sentence, why per-step latency grows with position.
