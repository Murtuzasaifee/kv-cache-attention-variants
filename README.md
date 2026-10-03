# KV Cache & Attention Variants (Module 03)

10-session course module: Naive Decoding -> KV Cache Memory Math -> MHA -> MQA -> GQA -> MLA -> RoPE -> FlashAttention -> PyTorch SDPA -> PagedAttention/vLLM.

Standalone: no external prerequisite repo required — MHA is built from scratch in `src/kv_cache_variants/attention/mha.py`.

## Setup

```bash
uv sync --extra dev
```

Mac-first: runs on CPU/MPS by default (`sdpa_backends.default_device()`). No CUDA required for sessions 1-7 and 9; sessions 8 and 10 cover FlashAttention and PagedAttention/vLLM conceptually, with optional GPU experiments on your own hardware.

## Layout

- `src/kv_cache_variants/` — source of truth, all runnable modules
- `src/kv_cache_variants/lessons/` — one rich-terminal walkthrough script per session, reusing the modules above (no reimplemented math)
- `teaching_notebooks/` — full zero-to-advanced lecture notebooks (00 foundations through 04 MQA), detailed explanations, Mermaid diagrams, worked examples
- `docs/`, `mkdocs.yml` — MkDocs (Material) site rendering the module as browsable pages, including 12 interactive diagrams (`docs/diagrams/`) — `uv sync --extra docs && uv run mkdocs serve`

## Run

```bash
uv run ruff check src/                        # lint
uv run jupyter nbconvert --to notebook --execute teaching_notebooks/00_foundations.ipynb

# rich terminal walkthrough, per session (session01 ... session10)
uv run python -m kv_cache_variants.lessons.session01_naive_decoding
```
