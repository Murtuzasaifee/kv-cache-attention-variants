# Jupyter Teaching Notebooks — Implementation Plan

Zero-to-advanced teaching notebooks for the KV-Cache & Attention Variants module. These are separate from the existing `notebooks/` folder (10 thin wrappers around `src/`, used for dev sanity-checks tied to `assignments/`/`slides/`). These new notebooks live in **`teaching_notebooks/`** and are the actual lecture material — full explanations, Mermaid diagrams, many worked examples — to be taught to students *before* handing them the assignments.

Each notebook is **standalone**: runnable top-to-bottom on a fresh kernel with no dependency on any other notebook having been run. Every notebook imports only from `src/kv_cache_variants/` (already implemented and tested — no source changes needed) plus `torch`, `rich`, `matplotlib`, and stdlib.

## Folder

```
teaching_notebooks/
├── 00_foundations.ipynb
├── 01_naive_decoding.ipynb
├── 02_kv_cache_memory_math.ipynb
├── 03_mha_recap.ipynb
├── 04_mqa.ipynb
├── 05_gqa.ipynb
├── 06_mla.ipynb
├── 07_rope.ipynb
├── 08_flashattention.ipynb
├── 09_pytorch_sdpa.ipynb
└── 10_pagedattention_vllm.ipynb
```

## Common structure applied to every notebook

Each notebook follows this markdown/code cell rhythm per concept, repeated for every sub-topic in the notebook:

1. **Concept intro** (markdown) — plain-language statement of the idea, assuming nothing from earlier notebooks except what's briefly recapped.
2. **Intuition / analogy** (markdown) — a non-technical mental model before any math.
3. **Math** (markdown, LaTeX) — the formal definition/derivation.
4. **Mermaid diagram** (markdown code fence, ```mermaid```) — visual structure/flow/sequence, rendered natively by JupyterLab/nbviewer.
5. **Code example(s)** (code cells) — at least 2 per concept, progressively harder (toy shapes first, realistic model-sized shapes second), using `rich.Table`/`Panel` output for readability, importing straight from `src/kv_cache_variants/`.
6. **"Try it yourself"** (code cell with an inline prompt comment) — a one-parameter tweak (e.g. change `num_heads`, `seq_len`, `block_size`) for the student to rerun and observe.
7. **Recap + bridge** (markdown) — one paragraph tying the result to what's coming next.

Every notebook ends with a markdown cell: "You are now ready for `assignments/0N_*/task.md`."

Reused source (no reimplementation, per repo's existing reuse-first pattern):
`memory_calc.py`, `naive_decode.py`, `attention/{mha,mqa,gqa,mla}.py`, `rope.py`, `sdpa_backends.py`, `bench.py`, and the rich-terminal reference logic in `src/kv_cache_variants/lessons/session01..10_*.py` (ported into notebook cells + narrative, not copied verbatim as scripts).

---

## Notebook 00 — Foundations (new; not in the original 10-session spec, added because students start at zero)

**Goal:** everything needed to understand attention before session 1 even mentions a KV cache.

Sections:
1. What is a token, what is an embedding — toy 6-word vocabulary, embedding lookup table by hand.
2. Self-attention intuition — "each word asks a question (Q), every word offers a label (K) and a value (V)."
3. Scaled dot-product attention derived step by step: `scores = QK^T/√d_k` → mask → softmax → `·V`, first in pure Python on 3 numbers, then in `torch`.
4. Causal masking — why generation can't look ahead; upper-triangular mask visualized.
5. Autoregressive generation loop — one token at a time, feeding output back as input.
6. Where this fits in a transformer decoder block (attention + FFN + residual + norm, one paragraph, not deep-dived — full depth is out of scope, just enough to place KV-caching in context).

Worked examples (8-10 total): hand-traced 3-token attention with concrete numbers; embedding lookup for a 6-word sentence; causal mask for seq_len 4/8; softmax temperature effect demo; full `torch` forward pass on a toy sequence with printed intermediate shapes at every step.

Mermaid diagrams:
- Flow: `token → embedding → Q/K/V projection → attention scores → softmax → weighted sum → output`.
- Sequence diagram: causal mask blocking future positions per row.

---

## Notebook 01 — Naive Decoding

**Recap:** one sentence restating notebook 00's autoregressive loop.

Sections: why generating token N+1 without caching means recomputing attention over all N previous tokens from scratch; total work across a full generation is O(n²) in sequence length.

Reuses: `naive_decode.py` (`TinyCausalAttention`, `generate_naive`).

Examples: hand-trace what gets recomputed at decode steps 1, 2, 3 (a small table showing "tokens attended to" growing each step); a timing table across seq_len = 8/32/128/512 showing the quadratic blowup; complexity derivation `1+2+...+n = n(n+1)/2`.

Mermaid: sequence diagram — one lifeline per decode step, each step's box spanning the *entire* growing prefix (visually showing repeated work).

---

## Notebook 02 — KV Cache Memory Math

**Recap:** naive decoding recomputes K/V for old tokens every step — what if we just... kept them?

Sections: derive `kv_cache_bytes` formula from first principles — why only K and V are cached (not Q, not attention outputs); per-token bytes = `2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes`; scaling with batch size and sequence length.

Reuses: `memory_calc.py` (`kv_cache_bytes`, `human_bytes`).

Examples: worked byte-math for a 7B-, 13B-, 70B-parameter-shaped config at seq_len 2k/8k/32k; fp16 vs fp32 comparison; batch-size scaling table; reproduce a row of `reference_tables/model_config_memory_worksheet.md` by hand then verify against the function.

Mermaid: flowchart of formula inputs (`num_layers`, `num_kv_heads`, `head_dim`, `seq_len`, `dtype_bytes`) converging to `kv_cache_bytes`.

---

## Notebook 03 — Multi-Head Attention (MHA) Recap

**Recap:** one line — memory math told us *how big* the cache is; now the mechanics of *what* MHA caches and how it grows.

Sections: why multiple heads (each head learns a different relation); splitting `d_model` into `num_heads × head_dim` and merging back; prefill (processing the full prompt at once) vs. incremental decode (one new token, reusing cached K/V from previous steps).

Reuses: `attention/mha.py` (`MultiHeadAttention`).

Examples: shape-by-shape walkthrough of prefill (seq_len=5) followed by 5 incremental decode steps, printing Q/K/V/cache shapes at every step; visualize per-layer cache size growing by exactly one token's worth per step across a 4-layer toy model.

Mermaid: flowchart of MHA forward pass (input → split heads → scaled-dot-product per head → concat → output projection); state diagram `prefill → decode step → decode step → ...`.

---

## Notebook 04 — Multi-Query Attention (MQA)

**Recap:** MHA's cache cost scales with `num_kv_heads = num_heads` — what if we only kept one shared K/V head?

Sections: MQA derivation — all query heads attend against a single shared K/V head; direct cache-size consequence.

Reuses: `attention/mqa.py` (`MultiQueryAttention`), `memory_calc.py`.

Examples: side-by-side MHA vs. MQA cached K-head counts and total cache bytes across 3 model configs; discussion example of the quality/capacity tradeoff (why MQA can hurt quality — fewer distinct K/V subspaces).

Mermaid: fan-in diagram — N query heads converging onto 1 shared K/V head.

---

## Notebook 05 — Grouped-Query Attention (GQA)

**Recap:** MHA and MQA are two extremes — GQA is the tunable middle ground.

Sections: GQA as generalization via `num_kv_groups`; show numerically that `num_kv_groups=1` reduces to MQA and `num_kv_groups=num_heads` reduces to MHA.

Reuses: `attention/gqa.py` (`GroupedQueryAttention`).

Examples: full sweep table (and matplotlib bar chart) across `num_kv_groups = [1, 2, 4, 8]` showing cache bytes/token; numeric equivalence check at the two boundary configs against `mqa.py`/`mha.py` outputs (`torch.allclose`).

Mermaid: grouped fan-in diagram — groups of query heads each sharing one K/V head.

---

## Notebook 06 — Multi-Head Latent Attention (MLA)

**Recap:** GQA reduces cache by sharing whole K/V heads; MLA instead compresses K/V into a low-rank latent vector per token.

Sections: DeepSeek-V2 MLA — down-project K/V into a small latent, cache only the latent, up-project at attention time; why RoPE doesn't commute with the down/up projections, motivating MLA's "decoupled RoPE" path (a small separate RoPE'd component alongside the compressed latent).

Reuses: `attention/mla.py`.

Examples: matched-cache-budget comparison table across MHA / GQA(8) / MQA / MLA at equal per-token cache bytes, comparing effective capacity; numeric check that incremental decode with the cached latent equals a full recompute forward pass.

Mermaid: flow — `K/V → down-project → cache latent → (decode) → up-project → attention`; separate small diagram for the decoupled-RoPE side path.

---

## Notebook 07 — Rotary Positional Embeddings (RoPE)

**Recap:** none of the previous notebooks discussed *position* — this is where the model learns token order.

Sections: rotation-matrix derivation — rotating Q/K pairs by an angle proportional to position so their dot product depends only on relative position; then context-extension strategies: linear/Position-Interpolation scaling, dynamic NTK scaling, YaRN's per-frequency ramp.

Reuses: `rope.py` (`build_rope_cache`, `apply_rotary`, `linear_scaled_rope_cache`, `dynamic_ntk_rope_cache`, `yarn_rope_cache`).

Examples: hand-rotate a single 2D vector by a fixed angle on paper, then reproduce in `torch`; verify rotation preserves vector norm; incremental single-token rotation at an offset matches full-sequence recompute; frequency-vs-position plots (matplotlib) comparing unscaled/linear/NTK/YaRN at the lowest-frequency dimension when extrapolating past the trained context length.

Mermaid: diagram of rotation angle growing with position per frequency pair; YaRN's frequency-ramp diagram (which frequencies get scaled vs. left alone).

---

## Notebook 08 — FlashAttention (concept, via SDPA backend selection)

**Recap:** all prior variants reduced *what's cached*; FlashAttention instead speeds up the attention computation itself without changing the math.

Sections: IO-awareness — the bottleneck is HBM/memory bandwidth, not FLOPs; tiling so the full seq×seq score matrix is never materialized; on CPU/MPS we can't run the fused CUDA kernel, so `torch.nn.attention.sdpa_kernel` backend forcing (math/mem-efficient/flash) is used to teach the concept locally, with a pointer to the Colab notebook for real GPU numbers.

Reuses: `sdpa_backends.py` (`available_backends`, `default_device`, `run_sdpa`), `bench.py` (`benchmark`).

Examples: check which SDPA backends are available on the current device; benchmark table (latency, throughput) across available backends at a realistic shape; discussion of why local CPU/MPS timings may look similar while the real gap appears on CUDA (link to `colab/session08_flashattention_colab.ipynb`).

Mermaid: HBM-vs-SRAM tiling diagram; backend-dispatch decision flowchart.

---

## Notebook 09 — Migrating to `torch.nn.functional.scaled_dot_product_attention`

**Recap:** we've been hand-rolling `softmax(QK^T/√d)V` since notebook 00 — SDPA fuses this into one dispatching op.

Sections: why migrate even though the math is identical (memory + speed, ties back to notebook 08); numerical-equivalence proof methodology.

Examples: local `manual_attention()` reference function compared against `F.scaled_dot_product_attention` across 3+ shapes with a pass/fail table (`torch.allclose`, max-abs-diff column); live migration exercise — take notebook 03's MHA manual-math cell and replace it with an SDPA call, showing identical output.

Mermaid: before/after diagram of the migration (hand-rolled ops box → single fused SDPA box).

---

## Notebook 10 — PagedAttention / vLLM

**Recap:** everything so far assumed one contiguous per-sequence cache buffer — PagedAttention changes the *allocation strategy*, not the attention math.

Sections: naive preallocation (worst-case `max_len` per request) wastes memory when actual generated lengths vary; block-table allocation (OS-virtual-memory-style paging) allocates fixed-size blocks on demand; pointer to `colab/session10_vllm_pagedattention_colab.ipynb` for real vLLM throughput numbers (GPU required, not reproducible locally).

Examples: CPU-only simulation (no vLLM needed) comparing naive vs. paged total bytes allocated vs. actually-used lower bound across a batch of requests with randomized lengths, reporting % memory saved; sweep `block_size` to show the waste-per-request bound (`< block_size` tokens wasted per request instead of `max_len - actual_length`).

Mermaid: block-table → physical KV blocks mapping diagram (redrawn version of the vLLM paper's core figure).

---

## Build & verification plan (for the follow-on implementation pass)

1. Generate all 11 `.ipynb` files via `nbformat`, following the structure above.
2. Execute each standalone: `uv run jupyter nbconvert --to notebook --execute --inplace teaching_notebooks/<name>.ipynb`, fresh kernel each time, confirming zero dependency on any other notebook's state.
3. `uv run ruff check` is not applicable to notebook cells directly, but any code cell logic beyond simple calls into `src/` should be kept minimal — no new source-level abstractions, since all attention/memory-math code already exists and is tested in `src/kv_cache_variants/`.
4. Manual pass: confirm every Mermaid block renders in JupyterLab, and that each notebook's final cell correctly points to the matching `assignments/0N_*/task.md`.
