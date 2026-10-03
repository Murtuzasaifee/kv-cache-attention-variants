# Assignment 09 — Migrate a Hand-Rolled Module to SDPA

## Objective
Show that `MultiHeadAttention` (Session 3, already SDPA-backed internally)
and a direct `F.scaled_dot_product_attention` call on the same Q/K/V
projections produce identical outputs — confirming the "migration" changes
nothing numerically, only the backend dispatch mechanism.

## Starter
See `starter.py`.

## Acceptance Criteria
- `test_migration_equivalence.py` passes.
