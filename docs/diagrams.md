# Interactive Diagrams

12 standalone Archify diagrams that span the module. Each opens in a viewer with pan/zoom, theme switch, search, focus, and export to PNG/SVG.

A landing page linking all 12 is at **[diagrams/index.html](diagrams/index.html)**.

## Curriculum Overview

- **[D1 — Curriculum Dependency](diagrams/D01_curriculum_dependency.html)** — which sessions depend on which; the failure-and-fix narrative across the module.

## Decode Mechanics

- **[D2 — Naive vs Cached Decode](diagrams/D02_naive_vs_cached_decode.html)** — per-step sequence showing what gets recomputed (naive) versus reused (cached).
- **[D3 — MHA Attention with KV Cache](diagrams/D03_mha_attention_cache.html)** — single decode step: Q, K, V projections, cache append, attention, output.
- **[D4 — KV Cache Memory Formula](diagrams/D04_kv_cache_memory_formula.html)** — `per_token = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes`, scaled by seq_len × batch_size.

## Attention Variants

- **[D5 — MHA / MQA / GQA / MLA Cache Comparison](diagrams/D05_variant_cache_comparison.html)** — at matched config and budget, where the compression comes from.
- **[D6 — GQA Group-Sharing Spectrum](diagrams/D06_gqa_spectrum.html)** — `num_kv_groups` as the single dial from MHA → MQA.
- **[D7 — MLA Latent Compression](diagrams/D07_mla_latent_compression.html)** — two cached items (latent, k_rope); decompression on demand.

## Positional Encoding

- **[D8 — RoPE Rotation Flow](diagrams/D08_rope_rotation.html)** — cos/sin tables computed from position, applied by `apply_rotary` with offset for incremental decode.
- **[D9 — RoPE Context Extension](diagrams/D09_rope_context_extension.html)** — Linear / Dynamic NTK / YaRN — three strategies for extending trained context length.

## Systems and Serving

- **[D10 — FlashAttention IO-Aware Tiling](diagrams/D10_flashattention_tiling.html)** — naive (HBM-bound) vs FlashAttention (SRAM tiles, streaming softmax).
- **[D11 — PyTorch SDPA Backend Dispatch](diagrams/D11_sdpa_backend_dispatch.html)** — auto-probe, FlashAttention-2, memory-efficient, cuDNN, math fallback, user override.
- **[D12 — PagedAttention Block-Table Allocation](diagrams/D12_pagedattention_blocks.html)** — naive per-request contiguous vs block-table paged, with physical block pool.

## Source Specs

Each diagram is a standalone, self-contained HTML artifact — all data is inlined, so the files work offline and deploy as-is.

## Note on Curriculum Order

The diagrams use the **approved reordered sequence** (RoPE before MLA, MHA Recap before memory math) for pedagogical reasons. The session nav above uses the **original module spec** numbering. When teaching, follow the diagram order; when referencing the sessions, follow the file numbers.
