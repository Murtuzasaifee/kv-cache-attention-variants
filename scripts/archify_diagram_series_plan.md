# Diagram Series Plan — Archify Artifacts for Module 03

**Status:** Stage 1 (planning only — no JSON or HTML generated yet)
**Toolkit:** Archify skill (interactive HTML diagrams from typed JSON specs)
**Output format:** One self-contained HTML file per diagram (~12 files total)
**Target acceptance:** Each diagram passes all 9 showcase artifact checks with 0 errors, plus browser evidence at 1440×900, 1600×1000, 1920×1080.

---

## Section 1 — Overview & Series Strategy

### Why a series, not a mega-diagram

The Archify renderer caps each artifact at ~12 primary nodes for legibility. A full 11-session module crammed into one HTML would either be illegible or fail the showcase validation. The right granularity is **one concept per diagram, ~12 diagrams total**, each answering a single concrete question. Together they form a visual narrative spanning the module.

### Conceptual threads to cover

The instructor's approved session order has four interleaved threads. The diagram series weaves through all of them:

| Thread | Sessions it spans | Diagrams involved |
|---|---|---|
| **Computational** — what gets computed each step | 00, 01, 02 | D1, D2, D4 |
| **Architectural** — how variants differ | 02, 04, 05, 07 | D3, D5, D6, D7 |
| **Positional** — where RoPE composes | 06, 07 | D8, D9 |
| **Systems** — production-grade serving | 08, 09, 10 | D10, D11, D12 |

### Use cases for the diagrams

1. **Live class projection** — open during a session to anchor the visual
2. **Docs site assets** — embed via MkDocs (drop the HTML into `docs/` and link)
3. **Student self-study** — standalone files opened from the repository
4. **Instructor reference** — quick visual index of the whole module

Each diagram serves all four. They are exportable (SVG/PNG via the Archify viewer's built-in export) so they can be re-used in slides if desired.

---

## Section 2 — Per-Diagram Specifications

Each diagram has the same shape: purpose, type, nodes, edges, visual approach, emphasis.

---

### D1 — Module Curriculum Dependency Graph

- **Type:** `workflow`
- **Sessions served:** 00–10 (whole module overview)
- **Purpose (ONE question):** "Which sessions depend on which?"
- **Primary nodes (~12):**
  1. `00_foundations`
  2. `01_naive_decoding`
  3. `02_mha_recap` (moved from old 03)
  4. `03_kv_cache_memory_math` (moved from old 02)
  5. `04_mqa`
  6. `05_gqa`
  7. `06_rope`
  8. `07_mla`
  9. `08_flashattention`
  10. `09_pytorch_sdpa`
  11. `10_pagedattention_vllm`
  12. *(reserved — could be a "Assignments" terminal node)*
- **Key edges:** Sequential primary path `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10`. Optional parallel edges: `03 → 06` (MHA Recap also enables RoPE), `06 → 07` (RoPE enables MLA's decoupled path).
- **Visual approach:** Vertical or horizontal main rail; each node is a session with its primary concept below the title.
- **Emphasis:** The "failure-and-fix" narrative — every session fixes what the previous one broke.

---

### D2 — Naive vs Cached Decode Timeline

- **Type:** `sequence`
- **Sessions served:** 00, 01
- **Purpose:** "Show that naive decode recomputes attention for every past token at every step, while cached decode reuses it."
- **Participants (~5 columns):**
  1. `prompt_tokens` (external / input)
  2. `model.forward` (the attention module)
  3. `kv_cache` (storage — appears only in the cached lane)
  4. `next_token_logits`
  5. `output_token`
- **Key messages (~2 per step):**
  - Naive lane: at each step, `prompt_tokens → model.forward → next_token_logits → output_token`. K/V recomputed every step.
  - Cached lane: first step computes and writes `model.forward → kv_cache`. Subsequent steps short-circuit the recomputation: `kv_cache → model.forward → next_token_logits → output_token`.
- **Visual approach:** Two parallel lanes side by side, "naive" and "cached", for 4 generation steps. The crossover point that makes the cached lane cheaper is the visual punchline.
- **Emphasis:** The divergence in work per step — naive's slope is steep, cached's is flat.

---

### D3 — MHA Attention with KV Cache Mechanism

- **Type:** `dataflow`
- **Sessions served:** 02
- **Purpose:** "Show what gets cached and what gets recomputed at each decode step."
- **Primary nodes (~10):**
  1. `token_embeddings`
  2. `Q_projection`
  3. `K_projection`
  4. `V_projection`
  5. `kv_cache` (a buffer that grows)
  6. `attention_scores`
  7. `causal_mask`
  8. `softmax`
  9. `weighted_sum`
  10. `output_projection`
- **Key edges:** The token embeddings feed Q, K, V projections. New K and V are appended to the cache. The attention scores consume **cached K (all)** + **new Q (1)** + **cached V (all)**. Output flows through the projection to the next layer.
- **Visual approach:** Top-down flow with a side-loop for the cache. The cache is visually distinguished (different border / color) to make it the focus. Notation: solid arrow = compute this step; dashed arrow = read from cache.
- **Emphasis:** K and V for old tokens are *read*, not computed. Q for the new token is the only new query.

---

### D4 — KV Cache Memory Formula Visualization

- **Type:** `architecture` (or `workflow` if treated as a calculator flow)
- **Sessions served:** 03
- **Purpose:** "What does each term in `KV_per_token = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes` mean?"
- **Primary nodes (~10):**
  1. `num_layers` (input parameter)
  2. `num_kv_heads` (variant-dependent: see D5)
  3. `head_dim`
  4. `2` (for K and V) — visualized as a multiplier node
  5. `dtype_bytes` (fp16 / bf16 / fp32)
  6. `per_token_bytes` (intermediate)
  7. `seq_len`
  8. `batch_size`
  9. `total_cache_bytes` (result)
  10. `human_readable` (KB/MB/GB formatted)
- **Key edges:** Inputs converge into `per_token_bytes`, which multiplies by `seq_len` and `batch_size`, then formats to human-readable. Side annotations call out which input changes with which attention variant.
- **Visual approach:** Layered boxes — inputs at top, intermediate `per_token_bytes` in middle, outputs at bottom. The variant-dependence of `num_kv_heads` is the key teaching moment.
- **Emphasis:** The single source of truth formula that every later variant's cache size is measured against.

---

### D5 — Attention Variant Comparison at Matched Cache Budget

- **Type:** `architecture`
- **Sessions served:** 04, 05, 07
- **Purpose:** "At the same `d_model`, `num_heads`, how does KV cache size differ across MHA / MQA / GQA / MLA?"
- **Primary nodes (4 variants + comparison substrate, ~10):**
  1. `MHA` block (caches `num_heads` separate K/V heads)
  2. `GQA: 8 groups` block
  3. `GQA: 2 groups` block
  4. `MQA` block (1 shared K/V head)
  5. `MLA: latent=64, rope=16` block
  6. `num_kv_heads` shared input (varies per variant)
  7. `kv_cache_bytes_calculator`
  8. `cache_size_MB` result node, with one bar per variant
  9. *(reserved for worked example label — e.g. "32L, 128 head_dim, 4096 ctx, fp16")*
  10. *(reserved for quality-vs-cache tradeoff annotation)*
- **Key edges:** Each variant block feeds the shared `kv_cache_bytes_calculator`. Calculator output feeds the bar chart result node.
- **Visual approach:** Top row of four variant cards with their internal head structure visible (e.g. MHA shows `num_heads` small dots, MQA shows 1 dot, GQA shows `num_kv_groups` dots). Bottom row: cache size comparison.
- **Emphasis:** The compression comes from different places (head sharing vs latent compression), but the comparison is apples-to-apples on cache bytes.

---

### D6 — GQA Group-Sharing Spectrum

- **Type:** `architecture` (or `workflow` if laid out left-to-right)
- **Sessions served:** 05
- **Purpose:** "How does GQA interpolate between MHA and MQA?"
- **Primary nodes (~10):**
  1. `MHA: num_kv_groups = num_heads` (anchor)
  2. `GQA-32: 32 groups`
  3. `GQA-16: 16 groups`
  4. `GQA-8: 8 groups`
  5. `GQA-4: 4 groups`
  6. `GQA-2: 2 groups`
  7. `MQA: num_kv_groups = 1` (anchor)
  8. *(reserved — shared `num_query_heads` input)*
  9. *(reserved — `repeat_interleave` expansion node)*
  10. *(reserved — cache size curve output)*
- **Key edges:** Horizontal arrangement showing the spectrum. Shared input feeds all variants. Each variant has an internal "expand to `num_query_heads`" node showing how the KV heads are replicated for attention.
- **Visual approach:** Linear spectrum with cache size shrinking left-to-right (or whatever direction matches the layout). MQA and MHA bookend the spectrum.
- **Emphasis:** GQA is the generalization point — `num_kv_groups` is the single dial that controls everything.

---

### D7 — MLA Latent Compression Path

- **Type:** `dataflow`
- **Sessions served:** 07
- **Purpose:** "How does MLA compress K and V into a low-rank latent, with a separate RoPE path?"
- **Primary nodes (~10):**
  1. `token_embedding`
  2. `kv_down_proj` (d_model → latent_dim)
  3. `latent_cache` (the cached tensor — visually highlighted)
  4. `k_up_proj` (latent_dim → num_heads × head_dim)
  5. `v_up_proj` (same)
  6. `k_rope_proj` (decoupled, head_dim=rope_head_dim)
  7. `q_rope_proj`
  8. `apply_rotary` node (separate for Q and K_rope)
  9. `attention_scores` (concatenated content + rope paths)
  10. `output_projection`
- **Key edges:** Embedding → kv_down_proj → latent_cache. Cache → k_up_proj / v_up_proj → attention. Embedding → q_rope_proj / k_rope_proj → apply_rotary → attention. The two paths reconverge at attention scores.
- **Visual approach:** Two parallel compression paths — content (latent) on one side, RoPE (rotated) on the other. They merge at the attention node. The `latent_cache` is the visual star — everything else is upstream/downstream of it.
- **Emphasis:** The cache stores `(latent, k_rope)` which together are much smaller than full per-head K and V. RoPE must go on `k_rope_proj` not on `latent` because rotation doesn't commute with low-rank projection.

---

### D8 — RoPE Rotation Flow

- **Type:** `dataflow`
- **Sessions served:** 06
- **Purpose:** "How does RoPE encode position via per-pair rotation?"
- **Primary nodes (~10):**
  1. `token_embedding`
  2. `position_id`
  3. `inv_freq` computation node
  4. `cos_table` (cached, shape `seq_len × head_dim/2`)
  5. `sin_table` (cached, same shape)
  6. `Q` (input to be rotated)
  7. `K` (input to be rotated)
  8. `apply_rotary_Q`
  9. `apply_rotary_K`
  10. `rotated_outputs` to attention
- **Key edges:** `position_id` → `inv_freq` → `cos_table` + `sin_table`. `Q`, `K` + `cos/sin` → `apply_rotary_*` → `rotated_outputs`.
- **Visual approach:** Left side computes the rotation tables (positional info). Right side applies them. The pairing of dimensions `(0, dim/2)`, `(1, dim/2 + 1)`, etc. could be visually called out as a paired inset.
- **Emphasis:** Position enters only through the cos/sin tables, not through Q/K directly. `offset` parameter for incremental decode.

---

### D9 — RoPE Context Extension Strategies

- **Type:** `architecture`
- **Sessions served:** 06
- **Purpose:** "How do linear / dynamic NTK / YaRN scaling extend context length?"
- **Primary nodes (~10):**
  1. `RoPE base=10000` (input common to all)
  2. `target_seq_len` (input)
  3. `trained_max_seq_len` (input)
  4. `Linear scaling`: interpolate all frequencies by `scale_factor`
  5. `Dynamic NTK`: stretch base, keep high frequencies intact
  6. `YaRN`: ramp between linear (low freq) and untouched (high freq)
  7. `cos/sin_table: linear`
  8. `cos/sin_table: NTK`
  9. `cos/sin_table: YaRN`
  10. `frequency_response_curves` (visualization node, one per strategy)
- **Key edges:** Shared inputs feed three strategy blocks. Each produces a modified `cos/sin_table`. All three feed a comparison view (e.g. frequency-vs-position plot).
- **Visual approach:** Three parallel branches. Each branch's internal transformation is summarized (e.g. "divide all angles by scale_factor"). A bottom panel compares frequency responses visually.
- **Emphasis:** Different strategies preserve different frequency ranges. YaRN is the most surgical.

---

### D10 — FlashAttention IO-Aware Tiling

- **Type:** `dataflow`
- **Sessions served:** 08
- **Purpose:** "Why is FlashAttention faster than naive attention, and what stays the same?"
- **Primary nodes (~10):**
  1. `Q matrix` (input)
  2. `K matrix` (input)
  3. `V matrix` (input)
  4. `HBM (slow, large)` storage — global memory
  5. `SRAM (fast, small)` storage — on-chip
  6. `naive: load full Q@K.T to HBM` — bottleneck path
  7. `FlashAttention: tile-Q loop` — tile + SRAM-resident softmax
  8. `flash: streaming softmax` — online normalization
  9. `output matrix` (same in both)
  10. `HBM read/write count` (Naive: O(n²) writes; Flash: O(n))
- **Key edges:** Naive path: all three → HBM → CPU/GPU scalar ops → HBM. Flash path: Q tiled → SRAM block-by-block over K, V → streaming softmax → output, never materializing the full `n × n` scores matrix in HBM.
- **Visual approach:** Two lanes side by side. Naive is wide (big HBM footprint). Flash is narrow and tall (many small SRAM tiles). Output node is shared.
- **Emphasis:** Mathematically identical output; only the data movement changes. This is the IO-awareness argument.

---

### D11 — PyTorch SDPA Backend Dispatch

- **Type:** `workflow`
- **Sessions served:** 09
- **Purpose:** "How does SDPA pick a backend, and what's the fallback chain?"
- **Primary nodes (~10):**
  1. `F.scaled_dot_product_attention(...)` call
  2. `Backend probe` (probe which kernel backends work)
  3. `FlashAttention-2 backend`
  4. `Memory-efficient backend`
  5. `cuDNN backend`
  6. `Math backend` (always-available fallback)
  7. `sdpa_kernel(backend=...)` user override path
  8. `Hardware check: CUDA?`
  9. `Hardware check: MPS / CPU?`
  10. `Dispatch decision` (terminal — which backend is selected)
- **Key edges:** Entry → probe (forks per-hardware) → ordered backend preference list → dispatch. User override branch bypasses auto-selection.
- **Visual approach:** Diamond gate pattern — entry, hardware check, ordered preference, fallback to math. The fallback path (math) is always reachable.
- **Emphasis:** The fallback IS the teaching point. Math always works regardless of hardware.

---

### D12 — PagedAttention Block-Table Allocation

- **Type:** `architecture`
- **Sessions served:** 10
- **Purpose:** "How does PagedAttention reduce fragmentation versus naive per-request allocation?"
- **Primary nodes (~10):**
  1. `Request 1: seq_len=512`
  2. `Request 2: seq_len=2048`
  3. `Request 3: seq_len=300` (odd length to show fragmentation)
  4. `Naive: per-request contiguous allocation` — waste highlighted
  5. `Block table: request_1 → [B1]`
  6. `Block table: request_2 → [B2, B3, B4]`
  7. `Block table: request_3 → [B5]`
  8. `Physical block pool` (16 blocks, fixed-size)
  9. `Waste under paging` (bounded by `block_size - 1`)
  10. `Waste under naive` (often 50%+ for short requests)
- **Key edges:** Three requests → their block tables → shared physical block pool. Side-by-side annotation showing waste in each scheme.
- **Visual approach:** Top: three request cards. Middle: two allocation schemes (naive contiguous vs paged). Bottom: physical memory pool with blocks colored by request.
- **Emphasis:** The waste shrinks from "fragmented per request" to "at most one block per request."

---

## Section 3 — Proposed Diagram Set (12 total)

| # | Title | Type | Sessions served |
|---|---|---|---|
| D1 | Module Curriculum Dependency | `workflow` | 00–10 |
| D2 | Naive vs Cached Decode Timeline | `sequence` | 00, 01 |
| D3 | MHA Attention with KV Cache | `dataflow` | 02 |
| D4 | KV Cache Memory Formula | `architecture` | 03 |
| D5 | Attention Variant Cache Comparison | `architecture` | 04, 05, 07 |
| D6 | GQA Group-Sharing Spectrum | `architecture` | 05 |
| D7 | MLA Latent Compression Path | `dataflow` | 07 |
| D8 | RoPE Rotation Flow | `dataflow` | 06 |
| D9 | RoPE Context Extension Strategies | `architecture` | 06 |
| D10 | FlashAttention IO-Aware Tiling | `dataflow` | 08 |
| D11 | PyTorch SDPA Backend Dispatch | `workflow` | 09 |
| D12 | PagedAttention Block-Table Allocation | `architecture` | 10 |

**Total: 12 diagrams.** This sits at the upper end of the showcase complexity budget but none individually exceed 12 primary nodes.

**Coverage check:**
- Every session 00–10 has at least one diagram
- All four source attention variants (MHA, MQA, GQA, MLA) appear in D5/D6/D7
- RoPE composition with each variant called out in D7/D8/D9
- Both memory (D4, D5, D6, D12) and compute (D3, D10, D11) threads covered

---

## Section 4 — File Organization Plan

```
/Users/sourangshupal/Downloads/kv-cache-attention-variants/
├── site/
│   └── archify/
│       ├── D01_curriculum_dependency.html
│       ├── D02_naive_vs_cached_decode.html
│       ├── D03_mha_attention_cache.html
│       ├── D04_kv_cache_memory_formula.html
│       ├── D05_variant_cache_comparison.html
│       ├── D06_gqa_spectrum.html
│       ├── D07_mla_latent_compression.html
│       ├── D08_rope_rotation.html
│       ├── D09_rope_context_extension.html
│       ├── D10_flashattention_tiling.html
│       ├── D11_sdpa_backend_dispatch.html
│       ├── D12_pagedattention_blocks.html
│       └── index.html             ← landing page with all 12 linked
├── scripts/
│   ├── archify_diagram_series_plan.md   ← this file
│   ├── archify_D01_curriculum_dependency.json
│   ├── archify_D02_naive_vs_cached_decode.json
│   └── ... (one JSON per diagram)
```

**Why `site/archify/`:**
- `site/` is already a recognized output directory in this repo (MkDocs builds into it but lives separately from the source under `docs/`)
- Self-contained HTML files work offline and can be served by any static host
- `index.html` lets students browse all 12 from one place
- JSON specs stay in `scripts/` next to other generation scripts (e.g. `build_teaching_notebooks.py`)

**Naming convention:**
- Files: `D{NN}_{snake_case_topic}.html`
- JSON: `archify_D{NN}_{snake_case_topic}.json`
- Display titles: human-readable (`D1: Module Curriculum`)

---

## Section 5 — Build Sequence & Dependencies

### Stage A — Pilot (Diagram D1 first)

D1 (curriculum dependency) is the simplest — one main path, no technical detail required, low risk of validation failures. Build it end-to-end:
1. Author JSON spec → `validate` → `deliver` → `visual-check`
2. If validation fails twice consecutively, stop and report diagnostics per archify rules
3. Once D1 passes, the artifact approach is validated; remaining 11 can proceed with confidence

### Stage B — Core mechanism diagrams (D2, D3, D4 next)

These establish the module's main narrative. D2's sequence type needs careful participant labeling (per archify rules). D3 is the canonical dataflow pattern. D4 is a calculator visualization.

Build D2, D3, D4 sequentially — they share conceptual foundations and reveal whether the diagramming style holds together for the rest.

### Stage C — Variant comparison cluster (D5, D6, D7)

These are the structurally richest. D5 is the keystone — it compares all four variants. D6 expands GQA in isolation. D7 details MLA's dataflow.

If any of these fails showcase validation, the architecture-type complexity limit may have been hit. Consider splitting D5 into D5a (MHA/MQA/GQA triad) and D5b (MLA latent approach) if so.

### Stage D — RoPE cluster (D8, D9)

D8 and D9 share inputs; they could be built in parallel by separate agents. The context-extension comparison in D9 is the more complex one.

### Stage E — Production systems cluster (D10, D11, D12)

These are the capstones. D10's IO-awareness visualization is the trickiest. D11 is a clean workflow. D12 is straightforward architecture.

Build D11 and D12 first as warm-up, then D10 once warmed up.

### Parallel-build opportunities

Once the pilot (D1) succeeds:
- D2 (sequence) and D8 (dataflow) can be built by different agents in parallel — different types, no shared inputs
- D3 (dataflow) and D9 (architecture) can be parallel
- D11 (workflow) and D12 (architecture) can be parallel

But each pair needs to land before their conceptual neighbors build. So while any two at a time can be parallel, the sequence is roughly: A (1) → B (2,3,4) → C (5,6,7) → D (8,9) → E (10,11,12).

### Total estimate

Per diagram (after pilot): ~5–8 tool invocations minimum (author → validate → repair → deliver → visual-check). For 11 remaining diagrams: ~55–88 tool invocations. Realistic single-agent throughput: 6–8 diagrams per session. So this is a **2-session job** at minimum, possibly 3.

---

## Section 6 — Validation Gates

Per the archify skill, every diagram must satisfy:

### Pre-delivery (authoring loop)

```bash
node bin/archify.mjs validate <type> <candidate.json> --quality showcase --json
```

**Acceptance:** All 9 artifact checks report 0 composition errors and 0 warnings. A 4-check receipt is basic validation, not showcase acceptance.

### Delivery (final freeze)

```bash
node bin/archify.mjs deliver <type> <candidate.json> <output.html> --quality showcase --json
```

**Acceptance:** Non-zero exit is never success. SHA-256 + byte counts reported for both specification and artifact.

### Post-delivery (browser evidence)

```bash
node bin/archify.mjs visual-check <output.html> --json
```

**Acceptance:** Run at 1440×900, 1600×1000, 1920×1080. The diagram must contain within the viewport (`scrollWidth ≤ innerWidth`, `scrollHeight ≤ innerHeight`) and remain readable at every checked size. For diagrams intended for large displays, also check 2048×1320.

**Honest disclaimer:** `visual-check` proves deterministic artifact checks and bounded browser behavior. It does not prove perceptual polish. Visual review requires the instructor or a vision-capable reviewer to look at each HTML and confirm clarity.

### Authoring rules I will follow

- `meta.quality_profile` must be exactly `"showcase"`
- Maximum 12 primary nodes per artifact
- One obvious main path; side branches leave the nearest main-path node
- Omit `meta.visual_preset` (default to `classic`); omit `meta.subtitle`; omit `meta.legend` unless needed
- Use the instructor's English (this is an English module)
- Use semantic component types when type is `architecture`: `backend`, `frontend`, `database`, `cloud`, `security`, `messagebus`, `external`
- Each relationship keeps its semantic label; the renderer auto-routes them

---

## Section 7 — Open Questions for Instructor

Before I start Stage 2 (pilot D1), I need a few answers. These change the plan materially.

### Q1 — Output format for the diagrams

**Options:**
- (a) **Standalone HTML only** — open in browser, view, pan/zoom, export to PNG/SVG via built-in viewer
- (b) **HTML + auto-exported SVG/PNG per diagram** — additional raster copies ready for slide embedding
- (c) **HTML + integration into MkDocs docs site** — diagrams live under `docs/`, linked from session pages

**Recommendation:** **(a)** for v1. Built-in export handles (b) manually if you ever need raster. (c) is a separate mkdocs nav decision.

### Q2 — Granularity

**Options:**
- (a) **12 diagrams as planned** — one concept each, ~12 nodes each, clean separation
- (b) **6 diagrams at higher density** — two concepts per diagram, ~20+ nodes each, more info per click
- (c) **24 diagrams at lower density** — every concept from a notebook gets its own diagram, smallest individual files

**Recommendation:** **(a)** as planned. The archify 12-node cap argues against (b). (c) doubles the work without much pedagogical gain since the notebook code cells already cover the detail.

### Q3 — Pilot diagram choice

**Options:**
- (a) **D1 (curriculum dependency)** — simplest, validates the workflow but is less interesting
- (b) **D5 (variant cache comparison)** — keystone diagram, high impact, but harder first attempt
- (c) **D11 (SDPA backend dispatch)** — clean workflow pattern, medium complexity, immediately useful for session 09

**Recommendation:** **(a) D1 first.** Once that passes, do (c) D11 next as the medium-difficulty gate before tackling the conceptually richer ones.

### Q4 — Audience for the diagrams

**Options:**
- (a) **Student-facing** — primary use is class projection; clarity for first-time learners is paramount
- (b) **Mixed (student + instructor reference)** — needs to work for both audiences; slightly denser
- (c) **Instructor-only** — denser diagrams acceptable; can assume prior context

**Recommendation:** **(a) student-facing.** The instructor will have the live demo and notes; the diagram on the projector only needs to work for learners.

### Q5 — Anything else to emphasize?

Open-ended. Examples that would sharpen the plan:
- A specific concept students historically get stuck on that needs a dedicated diagram
- A production model (Llama 3, DeepSeek, Qwen) you want explicitly referenced in D5 or D12
- Whether D12 (PagedAttention) should include a vLLM API call layer or stay pure mechanics
- Anything else

---

## Section 8 — Status

**This is Stage 1.** No JSON specs, no HTML files generated. Awaiting instructor answers to Q1–Q5 before Stage 2 (pilot D1).

Once approved, execution path is:
1. Pilot: D1 → validate → deliver → visual-check
2. If pilot passes: D2–D12 in the order from Section 5 (with parallelization where useful)
3. After all 12: build the `index.html` landing page linking them all
4. Final: commit `.json` specs to `scripts/`, HTML artifacts to `site/archify/`
