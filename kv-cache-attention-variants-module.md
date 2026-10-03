# Module: KV Cache & Attention Variants
**Course:** AI Foundations — Module 03 ("KV Cache & Advanced Attentions")
**Scope note:** This file covers only the "KV Cache & Attention Variants" section of Module 03. "Finetuning Transformer Architectures" is intentionally excluded per instruction. "Scaling Laws" is a separate module to be built later — not included here.
**Level:** Zero → Advanced
**Format:** Markdown module spec for Claude Code handoff — use this file to scaffold notebooks, slide decks, assignment repos, and grading rubrics.

---

## 0. Module Rationale (for instructor reference, not for students)

This module assumes completion of Tokenization Deep Dive and Transformers 101 — students already have a working single-head/multi-head/masked/cross-attention implementation and understand the vanilla transformer block. This module answers the question: *"the architecture works — now why does it fall over in production, and what did the field build to fix that?"*

Sequencing follows the actual failure-and-fix narrative of LLM inference: naive autoregressive decoding is wasteful → caching K/V fixes the compute waste but creates a memory problem → the memory problem is what motivated MQA/GQA/MLA → meanwhile the attention *compute* itself (not just the cache) has its own bottleneck, which FlashAttention fixes → PyTorch SDPA is where all of this lands as a usable API → PagedAttention/vLLM is what happens when you have to serve many concurrent requests, not just one. RoPE is placed alongside these because virtually every modern KV-cache-efficient architecture (LLaMA, DeepSeek, Qwen, etc.) uses it, and it interacts directly with how MLA compresses the cache.

**Primary anchor for this entire module:** Sebastian Raschka's 2026 body of work is the strongest available resource for this specific topic, more so than for earlier modules — it did not exist in this organized, current form even a year ago:
- "A Visual Guide to Attention Variants in Modern LLMs" (Mar 2026) — the conceptual spine for Sessions 4–7
  - https://magazine.sebastianraschka.com/p/visual-attention-variants
- "Understanding and Coding Self-Attention, Multi-Head Attention, Causal-Attention, and Cross-Attention in LLMs" — the code-level companion
  - https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
- "Coding the KV Cache in LLMs" — dedicated KV cache deep dive with implementation
  - https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- "The Big LLM Architecture Comparison" — cross-model reference table (which production models use which variant)
  - https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- LLM Architecture Gallery (continuously updated, 45+ visual model cards)
  - https://sebastianraschka.com/llm-architecture-gallery/
- `LLMs-from-scratch` GitHub repo — hands-on code folders per variant
  - https://github.com/rasbt/LLMs-from-scratch (see `ch04/04_gqa`, `ch04/05_mla`, `ch04/06_swa`, `ch04/08_deltanet`)

**Secondary anchor (systems/production layer):** Stanford CS336 (Spring 2026) Systems unit — use for the "why this matters at serving scale" framing and hardware/kernel perspective, not as the primary conceptual teaching resource.

---

## 1. Learning Objectives

By the end of this module, students should be able to:
1. Explain why naive autoregressive decoding recomputes redundant work at every generation step, and quantify the resulting waste.
2. Derive the KV cache memory formula and compute actual GPU memory requirements for a given model size and context length.
3. Implement a working KV cache in PyTorch on top of their Transformers 101 attention code.
4. Explain and implement Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), and articulate the quality/memory tradeoff each makes relative to MHA.
5. Explain Multi-Head Latent Attention (MLA) as a compression-based (not head-sharing-based) alternative to GQA, and explain why it can outperform GQA at large scale.
6. Explain and implement RoPE, and explain why it has become the default positional encoding in nearly all modern open-weight LLMs.
7. Explain FlashAttention's IO-awareness argument (why a mathematically-equivalent kernel can be dramatically faster).
8. Use PyTorch's `scaled_dot_product_attention` (SDPA) as the unified, hardware-dispatching attention API.
9. Explain PagedAttention's virtual-memory-inspired design and why it matters for multi-request serving (vLLM).
10. Given a target model size, context length, and hardware budget, recommend an appropriate attention variant and justify the choice quantitatively.

---

## 2. Prerequisites

- Completion of Transformers 101 (self-attention, multi-head attention, masking, positional encoding, cross-attention all assumed known).
- Working PyTorch `MultiHeadAttention`/`Head` implementation from that module — this module extends it directly rather than starting fresh.
- Basic familiarity with GPU memory concepts (VRAM, bytes-per-parameter) is helpful; taught inline in Session 2 if not.

---

## 3. Module Structure — Sessions, Readings, Videos, Assignments

### Session 1 — The Naive Decoding Problem
**Concepts:** autoregressive generation — one token at a time; why naive decoding recomputes K and V for every previous token at every single generation step; the resulting O(n²) redundant compute; framing this as the motivating problem for everything else in the module.

**Primary reading:**
- Michael Brenndoerfer, "KV Cache Explained: Efficient Attention for LLM Generation" — opens with exactly this framing before formalizing caching
  - https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization
- Garima Tripathi, "KV Cache Explained: Speeding Up Transformer Inference" — accessible framing of the speedup magnitude
  - https://medium.com/@garimatripathi0778/kv-cache-explained-speeding-up-transformer-inference-c446fee46e45

**Assignment mapping:** `assignments/01_naive_decoding/` — instrument the students' Transformers 101 decoder-only model: generate a 50-token continuation with no caching, log wall-clock time and FLOPs per generated token, and plot how per-token generation time grows as sequence length increases. This empirical result is the motivation for Session 2.

---

### Session 2 — KV Cache & Memory Math
**Concepts:** what gets cached (K, V per layer, per head) and why Q does not need caching; the KV cache size formula; computing real memory requirements for realistic model configurations; the linear-growth-with-context problem this creates.

**Primary reading:**
- Sebastian Raschka, "Coding the KV Cache in LLMs" — primary reading, includes working implementation
  - https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- Reference memory formula (use for worked examples in class):
  `KV_per_token = 2 × H × D × L × B` (H = attention heads, D = head dimension, L = transformer layers, B = bytes per element), `KV_cache_size = KV_per_token × context_length`
  — cross-check against: https://arxiv.org/pdf/2603.20397 (worked figures for LLaMA-2 variants at fp16 across context lengths, with VRAM limits marked)

**Instructor note:** build a live worked example using a model size Paul's students will recognize (e.g., a 7B-parameter LLaMA-style config) and have them compute KV cache size at 4K, 32K, and 128K context by hand before writing any code — this is what makes the "memory grows linearly and becomes the bottleneck" claim concrete rather than abstract.

**Assignment mapping:** `assignments/02_memory_math/` — given 4 model configurations (varying heads, layers, head dimension), students compute KV cache memory at three context lengths each and identify which configurations would exceed a given GPU's VRAM (after accounting for model weights). Then: implement the actual KV cache (append-only K/V buffers per layer) in their Session 1 model and re-run the Session 1 benchmark — generation should now show flat, not growing, per-token latency.

---

### Session 3 — Multi-Head Attention (MHA) Recap, Reframed for Inference
**Concepts:** quick recap of MHA from Transformers 101, now explicitly reframed around its inference cost — every query head has its own K/V head, so KV cache size scales directly with number of heads.

**Primary reading:**
- Sebastian Raschka, "A Visual Guide to Attention Variants in Modern LLMs," Section 1 (MHA) and 1.2–1.5 (historical background, self-attention internals, multi-head extension)
  - https://magazine.sebastianraschka.com/p/visual-attention-variants
- Companion code-level treatment: "Understanding and Coding Self-Attention, Multi-Head Attention, Causal-Attention, and Cross-Attention in LLMs"
  - https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

**Assignment mapping:** `assignments/03_mha_inference_cost/` — no new code; extend the Session 2 memory-math table to explicitly show "if we reduce KV heads while keeping query heads fixed, what happens to cache size?" This sets up the motivating question for Sessions 4 and 5.

---

### Session 4 — Multi-Query Attention (MQA)
**Concepts:** all query heads share a single K/V head; large memory savings; quality degradation tradeoff.

**Primary reading:**
- Raschka, "Visual Guide," MQA is introduced as the extreme case of the GQA spectrum (Section 2.2) — read that framing alongside the original MQA paper's motivation
  - Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (original MQA paper)
- Kantzuling, "Multi-head vs Multi-query vs Grouped-query attention" — clean side-by-side cache-size arithmetic
  - https://medium.com/@kantzuling0307/multi-head-vs-multi-query-vs-grouped-query-attention-6981715eb6ec

**Assignment mapping:** `assignments/04_mqa/` — modify the `MultiHeadAttention` module from Transformers 101 so all heads share one K/V projection; re-measure cache size using the Session 2 formula and confirm the reduction factor matches theory (cache size should shrink by a factor of H, the original head count).

---

### Session 5 — Grouped-Query Attention (GQA)
**Concepts:** GQA as the interpolation between MHA and MQA — query heads divided into groups, each group sharing one K/V head; the "uptraining" technique for converting existing MHA checkpoints to GQA cheaply; why GQA became the industry-standard default.

**Primary reading:**
- Raschka, "Visual Guide," Section 2 in full (GQA) — <cite index="85-1">explains GQA lets several query heads share the same key-value projections, making KV caching much cheaper without changing the overall decoder recipe very much</cite>, plus the "Why GQA Still Matters in 2026" discussion of the simplicity/quality tradeoff versus MLA
  - https://magazine.sebastianraschka.com/p/visual-attention-variants
- Hands-on code: `LLMs-from-scratch` repo, `ch04/04_gqa`
  - https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa
- Original paper: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
  - https://arxiv.org/abs/2305.13245

**Instructor note — production examples to cite live:** <cite index="85-1">Llama 3 8B, Qwen3 4B, Gemma 3 27B, Mistral Small 3.1 24B, SmolLM3 3B, and Tiny Aya 3.35B all use GQA as dense-model examples</cite>, giving students a direct "this is not academic, this is what's running in production right now" anchor.

**Assignment mapping:** `assignments/05_gqa/` — extend the MQA implementation from Session 4 into a general `GQA(num_query_heads, num_kv_groups)` module where `num_kv_groups=1` recovers MQA and `num_kv_groups=num_query_heads` recovers MHA; sweep across group counts and plot the cache-size-vs-group-count curve.

---

### Session 6 — Multi-Head Latent Attention (MLA)
**Concepts:** MLA as a fundamentally different approach — compressing what gets cached (a low-rank latent representation) rather than reducing how many K/V heads are cached; why this can preserve modeling quality better than GQA at large scale; the added implementation and serving complexity this introduces.

**Primary reading:**
- Raschka, "Visual Guide," Section 3 in full (MLA) — <cite index="85-1">explains MLA shrinks the cache by compressing what gets stored rather than by reducing how many K/Vs are stored by sharing heads,</cite> and covers the DeepSeek-V2 ablation results directly
  - https://magazine.sebastianraschka.com/p/visual-attention-variants
- Hands-on code: `LLMs-from-scratch` repo, `ch04/05_mla`
  - https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla
- Original paper: DeepSeek-V2 (2024) — https://arxiv.org/abs/2405.04434

**Instructor note — the sizing crossover point is a great discussion prompt:** <cite index="85-1">colleagues have observed that MLA only works well at a certain size — for smaller models (under ~100B parameters), GQA tends to work better, or is at least easier to tune and get right,</cite> and <cite index="85-1">the Sarvam team's own release is a clean natural experiment: their 30B model kept classic GQA while their 105B model switched to MLA.</cite> This is a good place to assign a short debate/position exercise: "at what scale would you switch, and why?"

**Assignment mapping:** `assignments/06_mla/` — implement a simplified MLA layer (low-rank K/V projection + reconstruction) following the `ch04/05_mla` reference; compare cache size and (on a small trained model) generation quality against the GQA implementation from Session 5 at matched cache-size budgets.

---

### Session 7 — RoPE Positional Encoding
**Concepts:** why RoPE replaced sinusoidal/learned positional encodings as the default; rotation-based encoding of relative position; why RoPE interacts directly with KV-cache-efficient architectures (it's applied to Q/K before caching, so it needs to compose correctly with MQA/GQA/MLA).

**Primary reading:**
- EleutherAI, "Rotary Embeddings: A Relative Revolution" — the canonical accessible explainer, <cite index="69-1">describing RoPE as a new type of position encoding that unifies absolute and relative approaches, and noting it either matches or surpasses other positional techniques across every architecture tested</cite>
  - https://blog.eleuther.ai/rotary-embeddings/
- Original paper: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  - https://arxiv.org/pdf/2104.09864
- Practical config reference: RoPE scaling strategies (linear, dynamic, YaRN) as implemented in Hugging Face's `rope_scaling` parameter — good bridge to long-context extension discussions
  - https://aiwiki.ai/wiki/rotary_position_embedding

**Assignment mapping:** `assignments/07_rope/` — implement RoPE (rotating Q/K pairs by position-dependent angle) and swap it in for the sinusoidal positional encoding built in Transformers 101, on the students' small GPT-style model; verify that relative-position behavior holds (e.g., attention score between two tokens depends on their distance, not their absolute positions) via a small unit test.

---

### Session 8 — FlashAttention
**Concepts:** the compute/memory-bandwidth distinction — attention's bottleneck isn't FLOPs, it's reading/writing the attention matrix to slow GPU memory; IO-aware tiling and kernel fusion; why this gives a 2–8x speedup for mathematically identical output.

**Primary reading:**
- Chaim Rand, "Increasing Transformer Model Efficiency Through Attention Layer Optimization" — accessible framing of what PyTorch SDPA's backends (including FlashAttention-2) actually do
  - https://medium.com/data-science/increasing-transformer-model-efficiency-through-attention-layer-optimization-fefa6f87b1d6
- ROCm/ AMD blog, "Accelerating Large Language Models with Flash Attention" — concrete benchmark numbers <cite index="64-1">showing FlashAttention provides a 2-8x speedup over naive attention, with the speedup increasing as sequence length grows due to naive SDPA's quadratic complexity</cite>
  - https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html
- Original paper: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
  - https://arxiv.org/abs/2205.14135

**Assignment mapping:** `assignments/08_flash_attention/` — benchmark the students' own naive attention implementation from Transformers 101 against PyTorch's built-in FlashAttention backend (via SDPA, see Session 9) at increasing sequence lengths; plot the growing speedup gap and connect it back to the IO-awareness argument rather than treating it as "just faster."

---

### Session 9 — PyTorch SDPA: The Unified Attention API
**Concepts:** `torch.nn.functional.scaled_dot_product_attention` as the production-standard entry point; automatic backend dispatch (FlashAttention-2, memory-efficient attention, math fallback, cuDNN); how this replaces hand-written attention modules in real codebases.

**Primary reading/tutorial (hands-on, do this one in-notebook):**
- PyTorch official tutorial, "Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA)" — <cite index="62-1">covers the sdpa_kernel context manager for asserting a specific backend, and building a CausalSelfAttention module that is torch-compilable</cite>
  - https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
  - Companion notebook download available directly on that page

**Assignment mapping:** `assignments/09_pytorch_sdpa/` — rewrite the students' `Head`/`MultiHeadAttention`/GQA modules to call `F.scaled_dot_product_attention` internally instead of manual Q@K.T/softmax/@V; use `sdpa_kernel()` to force each backend (math, efficient, flash) in turn and reproduce the Session 8 benchmark using the official API instead of a hand-rolled naive implementation.

---

### Session 10 — PagedAttention & vLLM (Capstone: Serving at Scale)
**Concepts:** the multi-request serving problem — KV cache per-request memory fragmentation; PagedAttention's virtual-memory/OS-paging-inspired design; non-contiguous physical memory blocks mapped via a block table; how this enables memory sharing across requests (parallel sampling, beam search, shared prefixes) and 2-4x throughput gains.

**Primary reading:**
- RunPod, "Introduction to vLLM and PagedAttention" — accessible on-ramp <cite index="82-1">explaining that the vLLM authors noticed existing inference engines only used 20-40% of available GPU memory, and applied the concept of OS-style memory paging to better utilize the KV cache</cite>
  - https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention
- Original paper: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" — <cite index="83-1">proposes storing the KV cache in fixed-size blocks that can be mapped to non-contiguous physical memory, and demonstrates 2-4x higher throughput than state-of-the-art systems like FasterTransformer and Orca at the same latency</cite>
  - https://arxiv.org/pdf/2309.06180
- Wentao's Blog summary — good structured recap for slide-building, covering <cite index="84-1">the block-level mapping table separating logical sequence positions from physical GPU blocks, and page-granular copy-on-write for sharing KV cache across parallel sampling, beam search, and shared prefixes</cite>
  - https://wentao.site/vllm_summary/

**Instructor note — production capstone framing:** this session is a natural place to pull in the CS336 Systems-unit framing of <cite index="93-1">inference as prefill and decode phases, with pruning, quantization, distillation, and speculative decoding as further speedup strategies</cite> — position it as "here's the broader toolbox PagedAttention sits inside," not something to teach in full depth in this module.

**Assignment mapping:** `assignments/10_pagedattention_vllm/` — no full reimplementation expected (block-table paging is a serving-infrastructure build, not a from-scratch exercise at this level). Instead: install and run vLLM locally or via a hosted notebook, serve the students' small trained model (or a small open-weight model) with vLLM, and benchmark concurrent-request throughput against a naive Hugging Face `generate()` loop at varying batch sizes — connect the observed throughput gap back to the paging/fragmentation argument from the readings.

---

## 4. Suggested Repo Structure (for Claude Code scaffolding)

```
kv-cache-attention-variants/
├── README.md                          # this module's syllabus, linked from here
├── slides/
│   ├── 01_naive_decoding.md
│   ├── 02_memory_math.md
│   ├── 03_mha_inference_cost.md
│   ├── 04_mqa.md
│   ├── 05_gqa.md
│   ├── 06_mla.md
│   ├── 07_rope.md
│   ├── 08_flash_attention.md
│   ├── 09_pytorch_sdpa.md
│   └── 10_pagedattention_vllm.md
├── assignments/
│   ├── 01_naive_decoding/
│   ├── 02_memory_math/
│   ├── 03_mha_inference_cost/
│   ├── 04_mqa/
│   ├── 05_gqa/
│   ├── 06_mla/
│   ├── 07_rope/
│   ├── 08_flash_attention/
│   ├── 09_pytorch_sdpa/
│   └── 10_pagedattention_vllm/       # vLLM serving benchmark, not from-scratch build
├── reference_tables/
│   └── model_config_memory_worksheet.md   # Session 2 worked examples (7B config at 4K/32K/128K context)
├── answer_keys/                       # instructor-only, not distributed to students
└── rubrics/
    └── grading_rubric.md              # Session 2 gate: correct cache-size arithmetic; Session 9 gate: SDPA speedup reproduced
```

## 5. Full Resource List (Reference)

| Resource | Type | Use |
|---|---|---|
| Raschka, "A Visual Guide to Attention Variants in Modern LLMs" | Article (Mar 2026) | Sessions 3, 4, 5, 6 — primary conceptual anchor |
| Raschka, "Understanding and Coding Self-Attention..." | Article + code | Session 3 |
| Raschka, "Coding the KV Cache in LLMs" | Article + code | Session 2 |
| Raschka, "The Big LLM Architecture Comparison" | Article | Cross-session reference for production model examples |
| Raschka, LLM Architecture Gallery | Living reference site | Cross-session reference |
| `rasbt/LLMs-from-scratch` repo (`ch04/04_gqa`, `05_mla`, `06_swa`, `08_deltanet`) | Code | Sessions 5, 6 assignment scaffolds |
| Ainslie et al., GQA paper (2023) | Paper | Session 5 |
| DeepSeek-V2 paper (2024) | Paper | Session 6 |
| EleutherAI, "Rotary Embeddings: A Relative Revolution" | Article | Session 7 |
| Su et al., RoFormer paper | Paper | Session 7 |
| Dao et al., FlashAttention paper | Paper | Session 8 |
| ROCm/AMD FlashAttention benchmark blog | Article | Session 8 |
| PyTorch official SDPA tutorial | Tutorial + notebook | Session 9 (hands-on anchor) |
| Kwon et al., PagedAttention/vLLM paper | Paper | Session 10 |
| RunPod, "Introduction to vLLM and PagedAttention" | Article | Session 10 |
| Stanford CS336 (Spring 2026), Systems unit | Course | Session 10 framing, production/serving context throughout |

---

## 6. Handoff Notes for Claude Code

- Sessions 3–6 build one running codebase (`MultiHeadAttention` → `MQA` → `GQA(num_kv_groups)` → `MLA`) — scaffold as incremental extensions of the Transformers 101 attention module, not standalone snippets, so Session 6's comparison assignment is a clean apples-to-apples test across variants at matched cache budgets.
- Session 2's memory-math worksheet should be built as a reusable calculator (function or small script), not just a one-off notebook cell — Sessions 4, 5, and 6 all reuse it.
- Session 9 is the practical bridge from "hand-rolled teaching code" to "what's actually in production" — treat it as the point where students' own modules get replaced by the real API call, and make sure the benchmark harness from Session 8 is written generically enough to be reused here without rewriting.
- Session 10 is explicitly scoped as a serving/benchmarking exercise, not a from-scratch build — do not over-invest Claude Code time trying to reimplement block-table paging; installing and benchmarking vLLM is the right level of effort.
- Do not reproduce Raschka's diagrams or DeepSeek/GQA paper figures verbatim in course slides — link to originals and the architecture gallery; build original worked examples (different model configs than the ones in the source articles) for in-class exercises.
- Flag to Paul: Session 6 (MLA) and Session 7 (RoPE) both connect directly to his in-progress SLM architecture research track — worth checking whether any worked examples from that research can be adapted into Session 2's memory-math worksheet for continuity.
