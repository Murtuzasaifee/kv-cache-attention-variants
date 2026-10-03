# Session 7 — RoPE

- Applied to Q/K BEFORE they're cached — cached K already has position baked in
- Rotation: pairs of dims rotated by an angle proportional to position x frequency
- Must compose with every cache variant: `apply_rotary(..., offset=past_len)` at each decode step
- Context extension strategies when seq_len exceeds training length:
  - Linear (Position Interpolation): compress positions by a fixed scale factor
  - Dynamic NTK: stretch the frequency base once past the trained length
  - YaRN: interpolate only low frequencies (long-range), leave high frequencies (local) alone
- Reference: `src/kv_cache_variants/rope.py`
