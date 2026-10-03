# KV Cache & Attention Variants

<p>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/package%20manager-uv-7c3aed?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/docs-Material%20for%20MkDocs-526cfe?logo=markdown&logoColor=white" alt="MkDocs">
  <img src="https://github.com/sourangshupal/kv-cache-attention-variants/actions/workflows/docs.yml/badge.svg" alt="Docs deploy">
</p>

> A zero-to-advanced course module on how modern LLMs actually serve text — from naive decoding, through the KV cache and every major attention-variant optimization (MQA, GQA, MLA, RoPE), to FlashAttention, PyTorch SDPA, and PagedAttention/vLLM.

---

## 📖 Description

Every token an LLM generates requires an attention computation over everything generated so far. Without optimization, that computation is re-done from scratch at every step and its cost grows linearly — this is **naive decoding**, and it is why inference is expensive.

This repository builds the full story from first principles, one idea at a time:

1. **Naive Decoding** — why recomputing attention at every step is wasteful
2. **KV Cache Memory Math** — the `per_token = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes` formula
3. **MHA Recap** — multi-head attention rebuilt from scratch, no shortcuts
4. **MQA** — multi-query attention: one KV head for everyone
5. **GQA** — grouped-query attention: `num_kv_groups` as the dial from MHA → MQA
6. **MLA** — DeepSeek-style latent compression of the KV cache
7. **RoPE** — rotary position embeddings, plus context-length extension (NTK, YaRN)
8. **FlashAttention** — IO-aware tiling and the online softmax
9. **PyTorch SDPA** — `F.scaled_dot_product_attention` and its backend dispatch
10. **PagedAttention / vLLM** — block-table paging for continuous batching

Everything is **standalone** — no external prerequisite repository. MHA is implemented from scratch in `src/kv_cache_variants/attention/mha.py`, and every session reuses those modules instead of re-implementing the math.

🌐 **Interactive documentation (with 12 explorable diagrams):** https://sourangshupal.github.io/kv-cache-attention-variants/

---

## 🗂️ Repository Structure

```
kv-cache-attention-variants/
├── src/kv_cache_variants/          # Source of truth — all runnable modules
│   ├── attention/                  # mha.py, mqa.py, gqa.py, mla.py
│   ├── lessons/                    # One rich-terminal walkthrough per session
│   ├── naive_decode.py             # Naive vs cached decode loop
│   ├── memory_calc.py              # KV cache memory formula
│   ├── rope.py                     # Rotary embeddings + context extension
│   ├── sdpa_backends.py            # SDPA backend probing/dispatch
│   └── bench.py                    # Benchmarking utilities
├── teaching_notebooks/             # Full lecture notebooks, sessions 00–04
│   ├── 00_foundations.ipynb
│   ├── 01_naive_decoding.ipynb
│   ├── 02_kv_cache_memory_math.ipynb
│   ├── 03_mha_recap.ipynb
│   └── 04_mqa.ipynb
├── docs/                           # MkDocs content — one page per session
│   └── diagrams/                   # 12 standalone interactive Archify diagrams
├── mkdocs.yml                      # MkDocs (Material) configuration
├── pyproject.toml                  # Project metadata + dependency groups
└── .github/workflows/docs.yml      # Auto-deploys the docs site to GitHub Pages
```

The 12 diagrams in `docs/diagrams/` are self-contained HTML — open them directly in a browser or explore them through the hosted docs. They cover the curriculum dependency graph, cache-shape comparisons across MHA/MQA/GQA/MLA, RoPE rotation, FlashAttention tiling, SDPA backend dispatch, and PagedAttention block tables.

---

## ⚙️ Setup

The project uses [**uv**](https://docs.astral.sh/uv/) as its package manager. If you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 🐍 Creating the virtual environment

You don't create the venv by hand — `uv sync` does it automatically. Running any command from the table below creates and manages `.venv/` in the project root for you:

```bash
uv sync --extra dev    # project + dev tools (ruff, mypy, pytest)
uv sync --extra docs   # project + docs tools (mkdocs, mkdocs-material)
```

Each group maps to the matching section in `pyproject.toml`, so students only install what they need.

### 📦 Installing the packages

```bash
uv sync          # core dependencies only (torch, numpy, matplotlib, jupyter, rich)
uv sync --all-extras   # everything at once
```

`uv.lock` is committed, so every install resolves to the exact same versions.

> 💻 **Mac-first:** runs on CPU/MPS out of the box (`sdpa_backends.default_device()`). No CUDA required — sessions covering FlashAttention and vLLM are presented conceptually with CPU-runnable stand-ins.

---

## 🚀 How to Execute

All commands run through `uv run …`, which automatically uses the project venv.

```bash
# ▶️ Rich terminal walkthrough, one per session (session01 … session10)
uv run python -m kv_cache_variants.lessons.session01_naive_decoding
uv run python -m kv_cache_variants.lessons.session05_gqa
uv run python -m kv_cache_variants.lessons.session10_pagedattention

# 📓 Execute a lecture notebook end-to-end
uv run jupyter nbconvert --to notebook --execute teaching_notebooks/00_foundations.ipynb

# 🔍 Lint the source
uv run ruff check src/

# 🌐 Serve the documentation site locally
uv sync --extra docs && uv run mkdocs serve
```

Then open http://127.0.0.1:8000 to browse the module with the interactive diagrams.

---

## 📚 Docs Site

Every push to `main` that touches `docs/`, `mkdocs.yml`, or `teaching_notebooks/` rebuilds and redeploys the site via GitHub Actions:

- **Site:** https://sourangshupal.github.io/kv-cache-attention-variants/
- **Diagram index:** https://sourangshupal.github.io/kv-cache-attention-variants/diagrams/

---

<p align="center">
  Built as a teaching module — every optimization is motivated by the failure of the one before it. 🧠⚡
</p>
