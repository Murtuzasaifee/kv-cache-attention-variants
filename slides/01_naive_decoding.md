# Session 1 — Naive Decoding

- Autoregressive generation: predict one token at a time, feed it back in
- Naive approach: re-run the FULL forward pass over the whole sequence-so-far at every step
- Cost: step t does O(t) attention work -> O(n^2) total work to generate n tokens
- What's NOT reused: K and V for every past position get recomputed from scratch each step
- Motivates: cache K/V once computed, reuse on later steps (Session 2+)
- Reference: `src/kv_cache_variants/naive_decode.py`
