# KV Cache & Attention Variants (Module 03)

10-session course module: Naive Decoding -> KV Cache Memory Math -> MHA -> MQA -> GQA -> MLA -> RoPE -> FlashAttention -> PyTorch SDPA -> PagedAttention/vLLM.

Standalone: no external prerequisite repo required — MHA is built from scratch in `src/kv_cache_variants/attention/mha.py`.

## Setup

```bash
uv sync --extra dev
```

Mac-first: runs on CPU/MPS by default (`sdpa_backends.default_device()`). No CUDA required for sessions 1-7 and 9. Sessions 8 and 10 have a Colab companion for the true GPU-only exercises (real FlashAttention-2 kernel, vLLM install) — see `colab/`.

## Layout

- `src/kv_cache_variants/` — source of truth, all runnable modules
- `src/kv_cache_variants/lessons/` — one rich-terminal walkthrough script per session, reusing the modules above (no reimplemented math)
- `notebooks/` — one thin notebook per session, imports from `src/`
- `teaching_notebooks/` — full zero-to-advanced lecture notebooks (00 foundations + 10 sessions), detailed explanations, Mermaid diagrams, worked examples — teach these before assigning `assignments/`. Plan: `jupyter-implementation.md`.
- `docs/`, `mkdocs.yml` — MkDocs (Material) site rendering `teaching_notebooks/` as browsable pages (`uv sync --extra docs && uv run mkdocs serve`)
- `colab/` — GPU-only exercises for Sessions 8 and 10
- `slides/`, `assignments/`, `answer_keys/`, `reference_tables/`, `rubrics/` — course content

## Run

```bash
uv run pytest tests/                          # unit tests
uv run ruff check src/                        # lint
uv run jupyter nbconvert --to notebook --execute notebooks/01_naive_decoding.ipynb
uv run jupyter nbconvert --to notebook --execute teaching_notebooks/00_foundations.ipynb

# rich terminal walkthrough, per session (session01 ... session10)
uv run python -m kv_cache_variants.lessons.session01_naive_decoding
```
