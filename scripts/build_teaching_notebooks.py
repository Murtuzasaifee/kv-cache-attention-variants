"""Generates teaching_notebooks/*.ipynb from cell content defined per notebook.

One-shot generator, not part of the runtime package — run once via
`uv run python scripts/build_teaching_notebooks.py`, per jupyter-implementation.md.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT_DIR = Path(__file__).resolve().parent.parent / "teaching_notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build(filename: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    path = OUT_DIR / filename
    with path.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 00 — Foundations
# ---------------------------------------------------------------------------

nb00 = [
    md("""
# Session 0 — Foundations: Tokens, Embeddings, Self-Attention

Zero-background start. Everything here builds toward one question every later
notebook answers differently: **what should a transformer remember about past
tokens while generating the next one?**

No prior notebook needed — this is where the course starts.
"""),
    md("""
## 1. Tokens and embeddings

A language model never sees text. It sees integers ("token ids") that a
tokenizer assigns to sub-words. Each integer is looked up in an **embedding
table** — a big matrix where row `i` is a learned vector representing token
`i`.

**Analogy:** a huge dictionary where every word has been replaced by its
coordinates on a map. Words used in similar contexts end up at nearby
coordinates.
"""),
    code("""
import torch

torch.manual_seed(0)

vocab = ["the", "cat", "sat", "on", "mat", "dog"]
vocab_size, d_model = len(vocab), 8

embedding_table = torch.randn(vocab_size, d_model)
sentence = ["the", "cat", "sat"]
token_ids = torch.tensor([vocab.index(w) for w in sentence])

embeddings = embedding_table[token_ids]
print("sentence:", sentence)
print("token ids:", token_ids.tolist())
print("embeddings shape:", tuple(embeddings.shape))  # (seq_len, d_model)
"""),
    md("""
## 2. Self-attention intuition

Every token produces three vectors from its embedding:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer, as a label?"
- **Value (V)** — "what do I actually contribute if picked?"

Each token's Query is compared against every other token's Key (dot product =
similarity). High similarity -> that token's Value contributes more to the
output.

**Analogy:** a library search. Your query is what you type in the search box.
Every book has a key (its catalogued subject tags). You get back a weighted
mix of book contents (values), weighted by how well each book's tags matched
your search.
"""),
    md("""
## 3. Scaled dot-product attention, by hand first

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V
$$

Dividing by $\\sqrt{d_k}$ keeps the dot products from growing too large as
dimension grows (which would make softmax too peaked / gradients too small).
"""),
    code("""
import math

# 3 tokens, head_dim = 4 -- small enough to print every number.
d_k = 4
Q = torch.randn(3, d_k)
K = torch.randn(3, d_k)
V = torch.randn(3, d_k)

scores = Q @ K.T / math.sqrt(d_k)
print("raw scores (3x3, one row per query token):\\n", scores)

weights = torch.softmax(scores, dim=-1)
print("\\nattention weights (each row sums to 1):\\n", weights)
print("row sums:", weights.sum(dim=-1))

output = weights @ V
print("\\noutput (3 tokens x d_k):\\n", output)
"""),
    md("""
## 4. Causal masking

Generation must not look at future tokens. Before softmax, we set scores for
"future" positions to $-\\infty$ so they get exactly 0 weight after softmax.
"""),
    code("""
from rich.console import Console
from rich.table import Table

console = Console()
seq_len = 5
mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

table = Table(title="causal mask (True = blocked / future position)")
table.add_column("query pos \\\\ key pos")
for j in range(seq_len):
    table.add_column(str(j), justify="center")
for i in range(seq_len):
    table.add_row(str(i), *["X" if mask[i, j] else "." for j in range(seq_len)])
console.print(table)
"""),
    code("""
causal_scores = torch.randn(seq_len, seq_len).masked_fill(mask, float("-inf"))
causal_weights = torch.softmax(causal_scores, dim=-1)
print("row 0 (only sees itself):", causal_weights[0].round(decimals=3))
print("row 4 (sees all 5 tokens):", causal_weights[4].round(decimals=3))
"""),
    md("""
## 5. Autoregressive generation

Generation is a loop: run the model, take the last position's output, turn it
into a token, append it, repeat. Each new token can attend to every token
before it (via the causal mask above).
"""),
    code("""
import torch.nn as nn

torch.manual_seed(0)
vocab_size, d_model = 50, 16
embed = nn.Embedding(vocab_size, d_model)
lm_head = nn.Linear(d_model, vocab_size, bias=False)


def toy_transform(x: torch.Tensor) -> torch.Tensor:
    # stand-in for "the rest of the transformer": identity for this toy demo.
    return x


prompt = torch.randint(0, vocab_size, (1, 3))
ids = prompt
for step in range(4):
    x = embed(ids)
    hidden = toy_transform(x)
    next_id = lm_head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=1)
    print(f"step {step}: generated token {next_id.item()}, sequence so far: {ids.tolist()}")
"""),
    md("""
## 6. Where this fits in a transformer decoder block

Each decoder layer is: `x -> self-attention -> add & norm -> feed-forward ->
add & norm -> next layer`. This course's 10 sessions live entirely inside the
self-attention sub-block — specifically, in what gets cached and how it's
computed. The feed-forward part never changes across any of the variants
below.
"""),
    md("""
```mermaid
flowchart LR
    A[token ids] --> B[embedding lookup]
    B --> C["Q, K, V projections"]
    C --> D["scores = Q K^T / sqrt(d_k)"]
    D --> E[causal mask]
    E --> F[softmax]
    F --> G["weighted sum of V"]
    G --> H[output projection]
```
"""),
    md("""
```mermaid
sequenceDiagram
    participant Q0 as query pos 0
    participant Q4 as query pos 4
    participant K0 as key pos 0
    participant K4 as key pos 4
    Q0->>K0: allowed
    Q0-xK4: blocked (future)
    Q4->>K0: allowed
    Q4->>K4: allowed (self)
```
"""),
    md("""
## Recap

You've now derived self-attention, causal masking, and the generation loop
from scratch. Every remaining notebook asks one question about this loop:
**as the sequence grows, what do we need to keep in memory to avoid
recomputing K and V for old tokens?** That's the KV cache — starting in
Notebook 1.

You are now ready to move to `01_naive_decoding.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 01 — Naive decoding
# ---------------------------------------------------------------------------

nb01 = [
    md("""
# Session 1 — Naive Decoding: The Cost of Not Caching

**Recap:** Notebook 0 built an autoregressive loop that re-ran the whole
model on the whole sequence at every step. This notebook makes the cost of
that explicit.
"""),
    md("""
## Why naive decoding is expensive

At decode step $t$, generating token $t+1$ requires attention over all $t$
previous tokens. If we don't cache anything, we recompute K and V for **all**
of those $t$ tokens from scratch, every single step — even though tokens
$1..t-1$ never change.
"""),
    code("""
import torch
import torch.nn as nn
from kv_cache_variants.naive_decode import TinyCausalAttention, generate_naive

torch.manual_seed(0)
vocab_size, d_model = 50, 32
embed = nn.Embedding(vocab_size, d_model)
lm_head = nn.Linear(d_model, vocab_size, bias=False)
attn = TinyCausalAttention(d_model)

prompt = torch.randint(0, vocab_size, (1, 4))
out = generate_naive(attn, prompt, embed, lm_head, max_new_tokens=8)
print("prompt length 4 + 8 new tokens ->", tuple(out.shape))
"""),
    md("""
## Hand-trace: what gets recomputed each step

Starting from a 4-token prompt, generating 3 more tokens:

| decode step | sequence length fed to model | tokens whose K/V get recomputed |
|---|---|---|
| 1 | 4 | 4 (all of them, again) |
| 2 | 5 | 5 (all of them, again) |
| 3 | 6 | 6 (all of them, again) |

Nothing here is reused — every step redoes work already done in the previous
step.
"""),
    code("""
from rich.console import Console
from rich.table import Table

console = Console()


class CountingAttention(TinyCausalAttention):
    def __init__(self, d_model: int = 32):
        super().__init__(d_model)
        self.tokens_processed_this_call = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.tokens_processed_this_call = x.shape[1]
        return super().forward(x)


counting_attn = CountingAttention(d_model)
prompt_len = 4
new_tokens = 5
ids = torch.randint(0, vocab_size, (1, prompt_len))

table = Table(title="naive decode: full recompute every step")
table.add_column("decode step")
table.add_column("sequence length fed to model", justify="right")
table.add_column("tokens recomputed (all of them)", justify="right")

for step in range(new_tokens):
    x = embed(ids)
    hidden = counting_attn(x)
    next_id = lm_head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=1)
    table.add_row(str(step + 1), str(counting_attn.tokens_processed_this_call),
                  str(counting_attn.tokens_processed_this_call))
console.print(table)
"""),
    md("""
## Timing blowup across sequence length

Total attention work across a full generation of `n` tokens (with no cache)
is $1 + 2 + \\dots + n = \\frac{n(n+1)}{2}$ — quadratic in `n`. With a cache,
each step only does $O(1)$ new work, so the total becomes $O(n)$.
"""),
    code("""
import time

table = Table(title="naive decode wall-clock time vs. total generated length")
table.add_column("total tokens generated")
table.add_column("time (s)", justify="right")
table.add_column("n(n+1)/2 (relative work units)", justify="right")

for n in [8, 32, 128, 256]:
    prompt = torch.randint(0, vocab_size, (1, 2))
    start = time.perf_counter()
    generate_naive(attn, prompt, embed, lm_head, max_new_tokens=n)
    elapsed = time.perf_counter() - start
    table.add_row(str(n), f"{elapsed:.4f}", str(n * (n + 1) // 2))
console.print(table)
"""),
    md("""
```mermaid
sequenceDiagram
    participant M as model
    Note over M: step 1: forward over tokens [1..4]
    M->>M: recompute K,V for tokens 1,2,3,4
    Note over M: step 2: forward over tokens [1..5]
    M->>M: recompute K,V for tokens 1,2,3,4,5 (1-4 redone!)
    Note over M: step 3: forward over tokens [1..6]
    M->>M: recompute K,V for tokens 1,2,3,4,5,6 (1-5 redone!)
```
"""),
    md("""
## Recap

Naive decoding wastes $O(n^2)$ total work by recomputing unchanged K/V every
step. The fix is obvious once stated: cache K and V for tokens already
processed, and only compute K/V for the *new* token each step.

You are now ready to move to `02_kv_cache_memory_math.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 02 — KV cache memory math
# ---------------------------------------------------------------------------

nb02 = [
    md("""
# Session 2 — KV Cache Memory Math

**Recap:** Notebook 1 showed naive decoding recomputes K/V for old tokens
every step. The fix — caching K and V — isn't free: it costs memory. This
notebook derives exactly how much.
"""),
    md("""
## Why only K and V get cached (not Q, not outputs)

At each decode step, only one new Query is needed (for the new token) — it's
immediately consumed and discarded. But that new Query must attend against
**every previous token's Key and Value**. Since old tokens' K/V never change,
they're the only thing worth keeping around.

$$
\\text{KV\\_per\\_token} = 2 \\times \\text{num\\_kv\\_heads} \\times \\text{head\\_dim} \\times \\text{num\\_layers} \\times \\text{dtype\\_bytes}
$$

$$
\\text{KV\\_cache\\_bytes} = \\text{KV\\_per\\_token} \\times \\text{seq\\_len} \\times \\text{batch\\_size}
$$

The factor of 2 is for K **and** V.
"""),
    code("""
from kv_cache_variants.memory_calc import kv_cache_bytes, human_bytes
from rich.console import Console
from rich.table import Table

console = Console()

# Llama-2-7B-ish config: 32 layers, 32 heads, head_dim 128, fp16.
configs = {
    "7B-ish": dict(num_layers=32, num_kv_heads=32, head_dim=128),
    "13B-ish": dict(num_layers=40, num_kv_heads=40, head_dim=128),
    "70B-ish": dict(num_layers=80, num_kv_heads=64, head_dim=128),
}

table = Table(title="KV cache size at seq_len=4096, fp16, batch=1 (full MHA, num_kv_heads=num_heads)")
table.add_column("model")
table.add_column("cache bytes", justify="right")
for name, cfg in configs.items():
    total = kv_cache_bytes(seq_len=4096, batch_size=1, dtype_bytes=2, **cfg)
    table.add_row(name, human_bytes(total))
console.print(table)
"""),
    code("""
table = Table(title="fp16 vs fp32, 7B-ish config, seq_len=4096")
table.add_column("dtype")
table.add_column("bytes/element")
table.add_column("total cache bytes", justify="right")
for dtype_name, dtype_bytes in [("fp16/bf16", 2), ("fp32", 4)]:
    total = kv_cache_bytes(num_layers=32, num_kv_heads=32, head_dim=128,
                            seq_len=4096, dtype_bytes=dtype_bytes)
    table.add_row(dtype_name, str(dtype_bytes), human_bytes(total))
console.print(table)
"""),
    code("""
table = Table(title="batch-size scaling, 7B-ish config, seq_len=4096, fp16")
table.add_column("batch_size")
table.add_column("total cache bytes", justify="right")
for batch in [1, 4, 16, 64]:
    total = kv_cache_bytes(num_layers=32, num_kv_heads=32, head_dim=128,
                            seq_len=4096, batch_size=batch, dtype_bytes=2)
    table.add_row(str(batch), human_bytes(total))
console.print(table)
console.print(
    "\\n[bold]Takeaway:[/bold] cache bytes scale LINEARLY with seq_len and "
    "batch_size, but multiplicatively with both together -- this is why "
    "serving many long-context requests concurrently is a memory problem, "
    "not just a compute problem."
)
"""),
    md("""
## Try it yourself

Reproduce a row of `reference_tables/model_config_memory_worksheet.md` by
hand: pick a config from that file, compute `KV_per_token` on paper, multiply
by `seq_len`, then verify against `kv_cache_bytes(...)` below.
"""),
    code("""
# Try it yourself: change these three numbers and recompute.
num_layers, num_kv_heads, head_dim = 32, 8, 128  # e.g. a GQA-8 config
seq_len = 8192

per_token = 2 * num_kv_heads * head_dim * num_layers * 2  # dtype_bytes=2
total = kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, dtype_bytes=2)
print(f"per-token: {human_bytes(per_token)}, total @ seq_len={seq_len}: {human_bytes(total)}")
assert total == per_token * seq_len
"""),
    md("""
```mermaid
flowchart TD
    A[num_layers] --> E[KV_per_token]
    B[num_kv_heads] --> E
    C[head_dim] --> E
    D[dtype_bytes] --> E
    E --> F["x seq_len"]
    F --> G["x batch_size"]
    G --> H[KV_cache_bytes]
```
"""),
    md("""
## Recap

Cache size is a simple product of 5 numbers. The one lever every attention
variant in this course pulls is `num_kv_heads` — MHA uses all of them, MQA
uses 1, GQA uses somewhere in between, MLA replaces heads with a compressed
latent dimension entirely.

You are now ready to move to `03_mha_recap.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 03 — MHA recap
# ---------------------------------------------------------------------------

nb03 = [
    md("""
# Session 3 — Multi-Head Attention (MHA) Recap

**Recap:** Notebook 2 gave us the memory formula. Now: the actual mechanics
of MHA's forward pass, prefill vs. incremental decode, and how the cache
grows in practice.
"""),
    md("""
## Why multiple heads

A single attention head computes one weighted average per query. Splitting
`d_model` into `num_heads` independent heads (each of size `head_dim =
d_model / num_heads`) lets the model attend to different *kinds* of
relationships in parallel (e.g. one head tracks syntax, another tracks
coreference) — then the heads' outputs are concatenated and mixed back
together by an output projection.
"""),
    code("""
import torch
from kv_cache_variants.attention.mha import MultiHeadAttention

torch.manual_seed(0)
d_model, num_heads = 32, 4
mha = MultiHeadAttention(d_model, num_heads)
print(f"d_model={d_model}, num_heads={num_heads}, head_dim={d_model // num_heads}")
"""),
    md("""
## Prefill vs. incremental decode

- **Prefill:** the full prompt is processed at once; `past_kv=None`; causal
  masking applies across the whole prompt.
- **Incremental decode:** one new token at a time; the *cached* K/V from all
  previous steps is passed in as `past_kv` and concatenated with the new
  token's K/V.
"""),
    code("""
from rich.console import Console
from rich.table import Table

console = Console()
x_prompt = torch.randn(1, 5, d_model)
out, kv = mha(x_prompt)  # prefill: past_kv=None
print("PREFILL")
print("  input shape:", tuple(x_prompt.shape))
print("  output shape:", tuple(out.shape))
print("  cached K shape:", tuple(kv[0].shape), "-- (batch, num_heads, seq_len, head_dim)")
print("  cached V shape:", tuple(kv[1].shape))

table = Table(title="incremental decode: cache grows by 1 token per step")
table.add_column("decode step")
table.add_column("new token seq shape")
table.add_column("cached K seq_len after step", justify="right")

for step in range(1, 6):
    next_x = torch.randn(1, 1, d_model)
    out, kv = mha(next_x, past_kv=kv)
    table.add_row(str(step), str(tuple(next_x.shape)), str(kv[0].shape[2]))
console.print(table)
"""),
    md("""
## Per-layer cache growth in a small multi-layer model
"""),
    code("""
num_layers = 4
layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
caches = [None] * num_layers

x = torch.randn(1, 3, d_model)  # 3-token prompt
for layer, layer_out_cache in enumerate(caches):
    x, caches[layer] = layers[layer](x)

table = Table(title="per-layer cache after prefill (seq_len=3) + 2 decode steps")
table.add_column("layer")
table.add_column("cached seq_len after prefill", justify="right")
table.add_column("cached seq_len after +2 decode steps", justify="right")

after_prefill = [c[0].shape[2] for c in caches]
for _ in range(2):
    x = torch.randn(1, 1, d_model)
    for layer in range(num_layers):
        x, caches[layer] = layers[layer](x, past_kv=caches[layer])
after_decode = [c[0].shape[2] for c in caches]

for layer in range(num_layers):
    table.add_row(str(layer), str(after_prefill[layer]), str(after_decode[layer]))
console.print(table)
"""),
    md("""
```mermaid
flowchart LR
    A[input x] --> B["split into num_heads"]
    B --> C["per-head scaled dot-product attention"]
    C --> D[concat heads]
    D --> E[output projection]
```
"""),
    md("""
```mermaid
stateDiagram-v2
    [*] --> Prefill
    Prefill --> Decode: past_kv = (K,V) from prompt
    Decode --> Decode: past_kv grows by 1 token
    Decode --> [*]
```
"""),
    md("""
## Recap

MHA caches `num_heads` full K/V heads per layer — the most expensive point
on the memory-vs-quality spectrum. Every following notebook is a variation on
"cache less by sharing or compressing K/V heads."

You are now ready to move to `04_mqa.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 04 — MQA
# ---------------------------------------------------------------------------

nb04 = [
    md("""
# Session 4 — Multi-Query Attention (MQA)

**Recap:** MHA caches `num_kv_heads = num_heads` full K/V heads. MQA asks:
what if every query head shared a *single* K/V head instead?
"""),
    md("""
## Derivation

All query heads still exist (still `num_heads` of them), but they all attend
against **one** shared Key head and **one** shared Value head. Cache size
drops by a factor of `num_heads` for the K/V portion, independent of how many
query heads there are.
"""),
    code("""
import torch
from kv_cache_variants.attention.mha import MultiHeadAttention
from kv_cache_variants.attention.mqa import MultiQueryAttention
from kv_cache_variants.memory_calc import kv_cache_bytes, human_bytes
from rich.console import Console
from rich.table import Table

console = Console()
torch.manual_seed(0)
d_model, num_heads = 32, 4

mha = MultiHeadAttention(d_model, num_heads)
mqa = MultiQueryAttention(d_model, num_heads)

x = torch.randn(1, 5, d_model)
_, mha_kv = mha(x)
_, mqa_kv = mqa(x)

print("MHA cached K shape:", tuple(mha_kv[0].shape), "-- num_heads distinct K heads")
print("MQA cached K shape:", tuple(mqa_kv[0].shape), "-- always exactly 1 K head")
"""),
    code("""
table = Table(title="MHA vs MQA cache bytes across model sizes (seq_len=4096, fp16)")
table.add_column("model")
table.add_column("MHA cache bytes (num_kv_heads=num_heads)", justify="right")
table.add_column("MQA cache bytes (num_kv_heads=1)", justify="right")
table.add_column("reduction factor", justify="right")

configs = {
    "7B-ish (32 layers, 32 heads, hd=128)": dict(num_layers=32, num_heads=32, head_dim=128),
    "13B-ish (40 layers, 40 heads, hd=128)": dict(num_layers=40, num_heads=40, head_dim=128),
    "70B-ish (80 layers, 64 heads, hd=128)": dict(num_layers=80, num_heads=64, head_dim=128),
}
for name, cfg in configs.items():
    mha_bytes = kv_cache_bytes(cfg["num_layers"], cfg["num_heads"], cfg["head_dim"], seq_len=4096)
    mqa_bytes = kv_cache_bytes(cfg["num_layers"], 1, cfg["head_dim"], seq_len=4096)
    table.add_row(name, human_bytes(mha_bytes), human_bytes(mqa_bytes),
                  f"{mha_bytes / mqa_bytes:.0f}x")
console.print(table)
"""),
    md("""
## The tradeoff

Fewer distinct K/V subspaces means less representational capacity for
attention patterns — MQA models can lose some quality relative to MHA at the
same parameter count, which is exactly why GQA (next notebook) exists as a
middle ground.
"""),
    md("""
```mermaid
flowchart LR
    Q1[query head 1] --> KV[shared K/V head]
    Q2[query head 2] --> KV
    Q3[query head 3] --> KV
    Q4[query head 4] --> KV
```
"""),
    md("""
## Try it yourself

Change `num_heads` above (e.g. to 8, 16, 32) and rerun the comparison table —
notice the MQA cache size never changes, only MHA's does.
"""),
    md("""
## Recap

MQA is the maximum-compression extreme: 1 shared K/V head no matter how many
query heads. GQA generalizes both MHA and MQA with a tunable group count.

You are now ready to move to `05_gqa.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 05 — GQA
# ---------------------------------------------------------------------------

nb05 = [
    md("""
# Session 5 — Grouped-Query Attention (GQA)

**Recap:** MHA (`num_kv_heads = num_heads`) and MQA (`num_kv_heads = 1`) are
two extremes. GQA makes `num_kv_groups` a tunable parameter in between.
"""),
    md("""
## Derivation

Query heads are split into `num_kv_groups` groups; every query head in a
group shares one K/V head. Only `num_kv_groups` K/V heads are ever cached.

- `num_kv_groups = 1` -> identical to MQA.
- `num_kv_groups = num_heads` -> identical to MHA.
"""),
    code("""
import torch
from kv_cache_variants.attention.gqa import GroupedQueryAttention
from kv_cache_variants.attention.mha import MultiHeadAttention
from kv_cache_variants.attention.mqa import MultiQueryAttention
from kv_cache_variants.memory_calc import kv_cache_bytes, human_bytes
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

console = Console()
torch.manual_seed(0)
d_model, num_heads = 32, 8

table = Table(title="GQA cache size sweep, num_heads=8, seq_len=4096")
table.add_column("num_kv_groups")
table.add_column("cache bytes/layer", justify="right")
group_counts = [1, 2, 4, 8]
bytes_per_group = []
for groups in group_counts:
    gqa = GroupedQueryAttention(d_model, num_heads, groups)
    total = kv_cache_bytes(num_layers=1, num_kv_heads=groups, head_dim=d_model // num_heads, seq_len=4096)
    bytes_per_group.append(total)
    table.add_row(str(groups), human_bytes(total))
console.print(table)

plt.figure(figsize=(5, 3))
plt.bar([str(g) for g in group_counts], bytes_per_group)
plt.xlabel("num_kv_groups")
plt.ylabel("cache bytes/layer")
plt.title("GQA cache size vs. num_kv_groups (num_heads=8)")
plt.show()
"""),
    md("""
## Numeric equivalence check

At the two boundary configs, GQA's output must exactly match MQA and MHA
respectively (same weights aren't shared here, so we check *shape*
equivalence and cached-head-count equivalence, not numerical output
equality, since these are separately-initialized modules).
"""),
    code("""
gqa_as_mqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups=1)
gqa_as_mha = GroupedQueryAttention(d_model, num_heads, num_kv_groups=num_heads)

x = torch.randn(1, 5, d_model)
_, kv_mqa_equiv = gqa_as_mqa(x)
_, kv_mha_equiv = gqa_as_mha(x)

print("GQA(num_kv_groups=1) cached K heads:", kv_mqa_equiv[0].shape[1], "== MQA's 1 head")
print("GQA(num_kv_groups=num_heads) cached K heads:", kv_mha_equiv[0].shape[1], f"== MHA's {num_heads} heads")
assert kv_mqa_equiv[0].shape[1] == 1
assert kv_mha_equiv[0].shape[1] == num_heads
"""),
    md("""
```mermaid
flowchart LR
    subgraph Group1
        Q1[query head 1] --> KV1[shared K/V head A]
        Q2[query head 2] --> KV1
    end
    subgraph Group2
        Q3[query head 3] --> KV2[shared K/V head B]
        Q4[query head 4] --> KV2
    end
```
"""),
    md("""
## Try it yourself

Change `num_heads` and the `group_counts` list above (e.g. `num_heads=16`,
groups `[1, 2, 4, 8, 16]`) and rerun — the reduction factor between adjacent
group counts is always exactly 2x when group counts double.
"""),
    md("""
## Recap

GQA is the practical default in most modern open-weight models (e.g.
Llama-2-70B, Mistral) because it lets teams pick a cache-size/quality point
without committing to either MHA or MQA's extreme.

You are now ready to move to `06_mla.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 06 — MLA
# ---------------------------------------------------------------------------

nb06 = [
    md("""
# Session 6 — Multi-Head Latent Attention (MLA)

**Recap:** GQA reduces cache by sharing whole K/V heads across query heads.
MLA (DeepSeek-V2) takes a different approach entirely: compress K/V into a
small low-rank **latent** vector per token, and cache only that.
"""),
    md("""
## Derivation

- Down-project the input into a `latent_dim`-sized vector (much smaller than
  `num_heads * head_dim`).
- Cache only this latent (plus a small decoupled RoPE component — see
  below).
- At attention time, up-project the latent back into full per-head K and V.

Cache size becomes proportional to `latent_dim`, **independent of
`num_heads`** — a fundamentally different scaling than MHA/MQA/GQA.
"""),
    md("""
## Why RoPE needs a separate ("decoupled") path

RoPE rotates Q/K by an angle depending on absolute position, then relies on
that rotation surviving into the dot product. But MLA's K comes from
up-projecting a compressed latent — the down/up projection is a *learned
linear map*, and rotation does not commute with an arbitrary linear map
applied afterward. Rotating post-projection would not produce the same
relative-position property RoPE relies on.

MLA's fix: compute a small, separate ("decoupled") RoPE'd key/query pair
directly from the raw input `x` (not from the compressed latent), of a much
smaller dimension (`rope_head_dim`), and concatenate it onto the
content-based K/Q at attention time. This decoupled piece is cached
alongside the latent, uncompressed but tiny.
"""),
    code("""
import torch
from kv_cache_variants.attention.mla import MultiHeadLatentAttention
from kv_cache_variants.attention.mha import MultiHeadAttention
from kv_cache_variants.attention.gqa import GroupedQueryAttention
from kv_cache_variants.attention.mqa import MultiQueryAttention
from kv_cache_variants.rope import build_rope_cache
from kv_cache_variants.memory_calc import kv_cache_bytes, human_bytes
from rich.console import Console
from rich.table import Table

console = Console()
torch.manual_seed(0)
d_model, num_heads, latent_dim, rope_dim = 32, 4, 8, 8
head_dim = d_model // num_heads

mla = MultiHeadLatentAttention(d_model, num_heads, latent_dim, rope_dim)
cos, sin = build_rope_cache(rope_dim, max_seq_len=16)
x = torch.randn(1, 5, d_model)
out, cache = mla(x, rope=(cos, sin))
print("MLA output shape:", tuple(out.shape))
print("cached latent shape:", tuple(cache[0].shape), "-- (batch, seq_len, latent_dim), independent of num_heads!")
"""),
    code("""
seq_len = 4096
mha_bytes = kv_cache_bytes(num_layers=1, num_kv_heads=num_heads, head_dim=head_dim, seq_len=seq_len)
gqa8_bytes = kv_cache_bytes(num_layers=1, num_kv_heads=2, head_dim=head_dim, seq_len=seq_len)
mqa_bytes = kv_cache_bytes(num_layers=1, num_kv_heads=1, head_dim=head_dim, seq_len=seq_len)
mla_bytes = kv_cache_bytes(num_layers=1, num_kv_heads=1, head_dim=latent_dim + rope_dim, seq_len=seq_len)

table = Table(title=f"cache bytes/layer @ seq_len={seq_len}, matched d_model={d_model}, num_heads={num_heads}")
table.add_column("variant")
table.add_column("cache bytes/layer", justify="right")
for name, total in [("MHA", mha_bytes), ("GQA (groups=2)", gqa8_bytes), ("MQA", mqa_bytes), ("MLA", mla_bytes)]:
    table.add_row(name, human_bytes(total))
console.print(table)
"""),
    md("""
## Incremental decode equivalence check

Decoding with the cached latent (`past_kv`) must produce cache growth
identical to what a fresh call with the full concatenated sequence would
produce.
"""),
    code("""
next_x = torch.randn(1, 1, d_model)
out2, cache2 = mla(next_x, past_kv=cache, rope=(cos, sin))
print("after 1 decode step, cached latent shape:", tuple(cache2[0].shape))
assert cache2[0].shape[1] == cache[0].shape[1] + 1
print("cache grew by exactly 1 token -- OK")
"""),
    md("""
```mermaid
flowchart LR
    X[input x] --> D[down-project]
    D --> L["cache latent (small!)"]
    L --> UK[up-project to K]
    L --> UV[up-project to V]
    X --> QR["decoupled RoPE q/k (small, cached separately)"]
    UK --> ATT[attention]
    UV --> ATT
    QR --> ATT
```
"""),
    md("""
## Recap

MLA decouples cache size from `num_heads` entirely — a different axis of
compression than GQA/MQA's head-sharing. This is what lets DeepSeek-V2/V3
serve very long contexts cheaply while keeping many query heads for quality.

You are now ready to move to `07_rope.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 07 — RoPE
# ---------------------------------------------------------------------------

nb07 = [
    md("""
# Session 7 — Rotary Positional Embeddings (RoPE)

**Recap:** none of the attention variants so far encoded token *position* --
attention as derived in Notebook 0 is permutation-invariant without it. RoPE
is how position enters the picture, and it's the mechanism MLA's decoupled
path (Notebook 6) had to work around.
"""),
    md("""
## Derivation intuition

RoPE rotates each Q/K vector (in 2D subspace pairs) by an angle proportional
to its position. Because rotating both Q and K by their respective
positions, then taking a dot product, the result depends only on the
**difference** in positions (relative position) — not on either position
absolutely. This is a clean way to inject position without adding it as a
separate term.
"""),
    code("""
import math
import torch

# Hand-rotate a single 2D vector by 30 degrees, by hand, then verify with torch.
angle = math.radians(30)
v = torch.tensor([1.0, 0.0])
rotation_matrix = torch.tensor([
    [math.cos(angle), -math.sin(angle)],
    [math.sin(angle), math.cos(angle)],
])
rotated = rotation_matrix @ v
print("original vector:", v.tolist())
print("rotated 30 degrees:", rotated.tolist())
print("norm preserved:", torch.allclose(v.norm(), rotated.norm(), atol=1e-6))
"""),
    code("""
from kv_cache_variants.rope import build_rope_cache, apply_rotary

torch.manual_seed(0)
head_dim, seq_len = 16, 8
cos, sin = build_rope_cache(head_dim, seq_len)
q = torch.randn(1, 2, seq_len, head_dim)
rotated = apply_rotary(q, cos, sin)

print("rotation preserves vector norm (rotation is orthogonal):",
      torch.allclose(rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-4))
"""),
    md("""
## Incremental rotation matches full recompute

Rotating a single new token at `offset=seq_len` must match rotating the full
extended sequence and slicing the last position -- this is what makes RoPE
compatible with incremental decoding.
"""),
    code("""
cos2, sin2 = build_rope_cache(head_dim, seq_len + 1)
q_next = torch.randn(1, 2, 1, head_dim)
full = torch.cat([q, q_next], dim=2)
rotated_full = apply_rotary(full, cos2, sin2)
rotated_next = apply_rotary(q_next, cos2, sin2, offset=seq_len)
matches = torch.allclose(rotated_full[:, :, -1:, :], rotated_next, atol=1e-5)
print("incremental single-token rotation matches full recompute:", matches)
assert matches
"""),
    md("""
## Context-extension strategies

Models are trained at some `trained_max_len`. Running them at longer
sequences means RoPE must extrapolate to positions/frequencies never seen in
training, which degrades quality unless corrected:

- **Linear / Position Interpolation** (Chen et al., 2023): compress positions
  by a fixed scale factor so the model "sees" positions within its trained
  range.
- **Dynamic NTK** (community/EleutherAI, 2023): stretch the rotation
  frequency base itself, only once `seq_len` exceeds the trained context.
- **YaRN** (Peng et al., 2023): a per-frequency ramp — low frequencies
  (long-range) get linearly interpolated, high frequencies (local, short-
  range) are left untouched, avoiding the local-attention quality loss that
  pure linear scaling causes.
"""),
    code("""
from kv_cache_variants.rope import linear_scaled_rope_cache, dynamic_ntk_rope_cache, yarn_rope_cache
import matplotlib.pyplot as plt

trained_len, extended_len = 8, 32
linear_cos, _ = linear_scaled_rope_cache(head_dim, extended_len, scale_factor=extended_len / trained_len)
ntk_cos, _ = dynamic_ntk_rope_cache(head_dim, extended_len, trained_max_len=trained_len)
yarn_cos, _ = yarn_rope_cache(head_dim, extended_len, trained_max_len=trained_len,
                               scale_factor=extended_len / trained_len)
base_cos, _ = build_rope_cache(head_dim, extended_len)

plt.figure(figsize=(6, 4))
for name, table_cos in [("no scaling", base_cos), ("linear", linear_cos),
                         ("dynamic NTK", ntk_cos), ("YaRN", yarn_cos)]:
    plt.plot(table_cos[:, 0].numpy(), label=name)
plt.axvline(trained_len, linestyle="--", color="gray", label="trained_max_len")
plt.xlabel("position")
plt.ylabel("cos(angle), lowest-frequency dim")
plt.legend()
plt.title(f"RoPE scaling strategies: trained_len={trained_len} -> extended_len={extended_len}")
plt.show()
"""),
    md("""
```mermaid
flowchart TD
    A[position index] --> B["angle = position x frequency"]
    B --> C[rotate Q/K pair]
    C --> D["dot product depends only on relative position"]
```
"""),
    md("""
```mermaid
flowchart LR
    A[all frequencies] --> B{"YaRN ramp"}
    B -->|high frequency, local| C[unscaled]
    B -->|low frequency, long-range| D[linearly interpolated]
```
"""),
    md("""
## Recap

RoPE encodes relative position via rotation, and extension strategies adapt
it for longer-than-trained contexts. This is the position mechanism every
attention variant in this course (MHA/GQA/MQA/MLA) can be combined with.

You are now ready to move to `08_flashattention.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 08 — FlashAttention
# ---------------------------------------------------------------------------

nb08 = [
    md("""
# Session 8 — FlashAttention (concept, via SDPA backend selection)

**Recap:** every notebook so far changed *what* gets cached. FlashAttention
instead speeds up the attention *computation* itself, without changing the
math at all -- same output, less memory traffic.
"""),
    md("""
## IO-awareness: the real bottleneck is memory bandwidth, not FLOPs

Naively computing attention materializes the full `seq_len x seq_len` score
matrix in slow HBM (GPU global memory). FlashAttention tiles the computation
into blocks that fit in fast on-chip SRAM, never writing the full score
matrix to HBM. The FLOP count is roughly unchanged; the memory traffic drops
dramatically, which is what actually limits speed on modern GPUs.

Locally on CPU/MPS we can't run the real fused CUDA kernel, so this notebook
teaches the *concept* via PyTorch's SDPA backend selection
(`torch.nn.attention.sdpa_kernel`), which dispatches to different
implementations of the identical math. The real CUDA FlashAttention-2 kernel
benchmark lives in `colab/session08_flashattention_colab.ipynb` (GPU
required) -- local results here are backend/device dependent, so grade on
methodology, not absolute numbers.
"""),
    code("""
import torch
from kv_cache_variants.sdpa_backends import available_backends, default_device, run_sdpa
from kv_cache_variants.bench import benchmark
from rich.console import Console
from rich.table import Table

console = Console()
device = default_device()
working = available_backends()
print(f"device: {device}, working SDPA backends here: {working}")
"""),
    code("""
batch, heads, seq_len, head_dim = 2, 8, 1024, 64
q = k = v = torch.randn(batch, heads, seq_len, head_dim, device=device)

table = Table(title=f"SDPA backend benchmark, shape=({batch},{heads},{seq_len},{head_dim})")
table.add_column("backend")
table.add_column("mean latency (ms)", justify="right")
table.add_column("throughput (calls/s)", justify="right")
for backend in working:
    result = benchmark(run_sdpa, q, k, v, backend=backend, warmup=3, iters=10)
    table.add_row(backend, f"{result['mean_s'] * 1000:.3f}", f"{result['throughput_per_s']:.1f}")
console.print(table)
"""),
    md("""
## Try it yourself

Increase `seq_len` above (e.g. 2048, 4096) and rerun -- on a real CUDA GPU
the gap between `math` and `flash`/`efficient` backends widens as sequence
length grows, because the quadratic-memory `math` path suffers more as the
score matrix grows. On CPU/MPS you may see the backends converge or fall
back to `math` -- see the Colab notebook for real numbers.
"""),
    md("""
```mermaid
flowchart TB
    subgraph Naive
        A1[load Q,K,V from HBM] --> A2["compute full seq x seq scores in HBM"]
        A2 --> A3[softmax in HBM]
        A3 --> A4[write output to HBM]
    end
    subgraph FlashAttention
        B1["load Q,K,V tile into SRAM"] --> B2["compute partial scores in SRAM"]
        B2 --> B3["accumulate softmax in SRAM"]
        B3 --> B4["write only final output tile to HBM"]
    end
```
"""),
    md("""
```mermaid
flowchart LR
    A["F.scaled_dot_product_attention call"] --> B{sdpa_kernel context}
    B -->|CUDA + supported shape| C[flash backend]
    B -->|CUDA, unsupported shape| D[efficient backend]
    B -->|no GPU support| E[math backend, fallback]
```
"""),
    md("""
## Recap

FlashAttention is a systems optimization, not a math change -- output is
numerically identical to naive attention. Notebook 9 makes that equivalence
explicit for the general SDPA migration.

You are now ready to move to `09_pytorch_sdpa.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 09 — SDPA migration
# ---------------------------------------------------------------------------

nb09 = [
    md("""
# Session 9 — Migrating to `torch.nn.functional.scaled_dot_product_attention`

**Recap:** since Notebook 0 we've hand-rolled `softmax(QK^T/sqrt(d))V`.
`F.scaled_dot_product_attention` (SDPA) fuses this into one op that
dispatches to whichever backend (math/flash/efficient/cudnn, Notebook 8) is
fastest on the current hardware -- with **identical output**.
"""),
    md("""
## Why migrate if the math is identical

SDPA avoids materializing the full attention matrix in memory and avoids
Python-level overhead of separate matmul/softmax/matmul calls -- same
numbers, less memory, faster. "A migration that changes outputs is a bug,
not a valid solution" -- so the whole point of this notebook is proving
numerical equivalence, not proving speed (that's Notebook 8).
"""),
    code("""
import math
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

console = Console()


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    \"\"\"Hand-rolled reference: softmax(QK^T / sqrt(d_k)) V, exactly what SDPA fuses.\"\"\"
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if is_causal:
        seq = q.shape[-2]
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


torch.manual_seed(0)
table = Table(title="manual softmax(QK^T/sqrt(d))V  vs  F.scaled_dot_product_attention")
table.add_column("shape (batch, heads, seq, head_dim)")
table.add_column("max abs diff", justify="right")
table.add_column("numerically identical (atol=1e-5)?", justify="center")
for shape in [(1, 4, 8, 16), (2, 8, 32, 64), (1, 2, 128, 32)]:
    q, k, v = (torch.randn(shape) for _ in range(3))
    manual_out = manual_attention(q, k, v, is_causal=True)
    sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = (manual_out - sdpa_out).abs().max().item()
    identical = torch.allclose(manual_out, sdpa_out, atol=1e-5)
    table.add_row(str(shape), f"{diff:.2e}", "yes" if identical else "NO - bug")
console.print(table)
"""),
    md("""
## Live migration exercise

Take the Session-3-style manual MHA math and replace it with SDPA, verifying
the migrated version produces the same output as the hand-rolled version.
"""),
    code("""
d_model, num_heads, seq_len = 32, 4, 6
head_dim = d_model // num_heads
q = torch.randn(1, num_heads, seq_len, head_dim)
k = torch.randn(1, num_heads, seq_len, head_dim)
v = torch.randn(1, num_heads, seq_len, head_dim)

before = manual_attention(q, k, v, is_causal=True)
after = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("before/after migration identical:", torch.allclose(before, after, atol=1e-5))
"""),
    md("""
```mermaid
flowchart LR
    subgraph Before
        A1[matmul QK^T] --> A2[scale by 1/sqrt(d)]
        A2 --> A3[apply causal mask]
        A3 --> A4[softmax]
        A4 --> A5[matmul with V]
    end
    subgraph After
        B1["F.scaled_dot_product_attention(q, k, v, is_causal=True)"]
    end
```
"""),
    md("""
## Recap

SDPA is a drop-in, numerically-identical replacement for hand-rolled
attention math, and every attention variant module in this repo
(`mha.py`/`mqa.py`/`gqa.py`/`gqa.py`-based MQA/`mla.py`) already calls it
internally -- this notebook made that migration explicit and provable.

You are now ready to move to `10_pagedattention_vllm.ipynb`.
"""),
]

# ---------------------------------------------------------------------------
# 10 — PagedAttention / vLLM
# ---------------------------------------------------------------------------

nb10 = [
    md("""
# Session 10 — PagedAttention / vLLM

**Recap:** every notebook so far assumed one contiguous per-sequence cache
buffer. PagedAttention (Kwon et al., 2023, the vLLM paper) changes the
**allocation strategy** for that buffer, not the attention math itself.
"""),
    md("""
## The problem: naive worst-case preallocation

If a serving system doesn't know how long a request's output will be in
advance, the naive approach preallocates a buffer sized for `max_len` per
request. Since actual generated lengths vary a lot, most of that
preallocated memory goes unused -- this is internal fragmentation, and it
limits how many concurrent requests fit in GPU memory.
"""),
    md("""
## The fix: block-table allocation

Split the cache into fixed-size blocks (like OS virtual-memory pages).
Allocate blocks on demand as a sequence grows, tracked via a per-sequence
block table mapping logical positions to physical blocks. Waste per request
is now bounded by at most `block_size - 1` tokens, instead of
`max_len - actual_length`.
"""),
    code("""
import random
from kv_cache_variants.memory_calc import kv_cache_bytes, human_bytes
from rich.console import Console
from rich.table import Table

console = Console()
random.seed(0)
num_layers, num_kv_heads, head_dim, max_len, block_size = 24, 8, 128, 2048, 16

# A batch of concurrent requests with varying ACTUAL generated lengths --
# this variance is exactly what makes naive worst-case preallocation wasteful.
actual_lengths = [random.randint(50, max_len) for _ in range(32)]

naive_total = sum(kv_cache_bytes(num_layers, num_kv_heads, head_dim, max_len) for _ in actual_lengths)
paged_total = sum(
    kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len=-(-length // block_size) * block_size)
    for length in actual_lengths
)
actually_used_total = sum(kv_cache_bytes(num_layers, num_kv_heads, head_dim, length) for length in actual_lengths)

table = Table(title=f"{len(actual_lengths)} concurrent requests, max_len={max_len}, block_size={block_size}")
table.add_column("allocation strategy")
table.add_column("total bytes allocated", justify="right")
table.add_column("wasted vs. actually used", justify="right")
table.add_row("naive (preallocate max_len per request)", human_bytes(naive_total),
              human_bytes(naive_total - actually_used_total))
table.add_row("paged (block_size-rounded, per-request)", human_bytes(paged_total),
              human_bytes(paged_total - actually_used_total))
table.add_row("actually used (lower bound)", human_bytes(actually_used_total), "-")
console.print(table)
print(f"memory saved by paging: {(1 - paged_total / naive_total) * 100:.1f}% less allocated than naive")
"""),
    md("""
## Try it yourself: sweep `block_size`
""") ,
    code("""
table = Table(title="waste bound shrinks as block_size shrinks (finer-grained allocation)")
table.add_column("block_size")
table.add_column("paged total bytes", justify="right")
table.add_column("waste vs actually used", justify="right")
for bs in [8, 16, 32, 64, 128]:
    paged = sum(
        kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len=-(-length // bs) * bs)
        for length in actual_lengths
    )
    table.add_row(str(bs), human_bytes(paged), human_bytes(paged - actually_used_total))
console.print(table)
console.print(
    "\\n[bold]Tradeoff:[/bold] smaller block_size -> less waste, but more block-table "
    "bookkeeping overhead per sequence -- real vLLM defaults balance this."
)
"""),
    md("""
## Beyond this notebook

This is a CPU-only allocation-strategy simulation, not vLLM's actual CUDA
paged-attention kernel. The real vLLM install + throughput benchmark (vs.
naive HuggingFace `generate()`) is in
`colab/session10_vllm_pagedattention_colab.ipynb` (GPU required).
"""),
    md("""
```mermaid
flowchart LR
    subgraph "Logical view (per sequence)"
        L0[token 0-15] --> L1[token 16-31] --> L2[token 32-47]
    end
    subgraph "Physical KV blocks (shared pool)"
        P7[block 7]
        P2[block 2]
        P9[block 9]
    end
    L0 -.block table.-> P7
    L1 -.block table.-> P2
    L2 -.block table.-> P9
```
"""),
    md("""
## Recap -- the whole course, tied together

- **Notebook 0-1:** why caching K/V matters at all.
- **Notebook 2:** how to compute cache size.
- **Notebook 3-6:** four ways to shrink `num_kv_heads`/cache layout (MHA
  baseline, MQA, GQA, MLA).
- **Notebook 7:** how position enters via RoPE, and how to extend context.
- **Notebook 8-9:** how the attention computation itself gets fast (SDPA,
  FlashAttention).
- **Notebook 10:** how a serving system allocates cache memory across many
  concurrent requests.

You are now ready for the assignments in `assignments/`, starting with
`assignments/01_naive_decoding/task.md`.
"""),
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    build("00_foundations.ipynb", nb00)
    build("01_naive_decoding.ipynb", nb01)
    build("02_kv_cache_memory_math.ipynb", nb02)
    build("03_mha_recap.ipynb", nb03)
    build("04_mqa.ipynb", nb04)
    build("05_gqa.ipynb", nb05)
    build("06_mla.ipynb", nb06)
    build("07_rope.ipynb", nb07)
    build("08_flashattention.ipynb", nb08)
    build("09_pytorch_sdpa.ipynb", nb09)
    build("10_pagedattention_vllm.ipynb", nb10)


if __name__ == "__main__":
    main()
