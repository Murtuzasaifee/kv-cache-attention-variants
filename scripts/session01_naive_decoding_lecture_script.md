# Session 01 — Naive Decoding: Lecture Script

**Duration:** ~75 minutes
**Materials:** Jupyter notebook `teaching_notebooks/01_naive_decoding.ipynb`, projector
**Prerequisite:** Session 00 (Foundations)
**Source code reference:** `src/kv_cache_variants/naive_decode.py`

---

## PART 0 — Recap and Frame (5 min)

> **[Standing, before any slides]**

Last session we built the mechanics from scratch. Tokens become embeddings. Embeddings produce Q, K, V. Attention computes a weighted lookup: queries match keys, values are the result. Causal masking ensures no peeking ahead. And generation is a loop — embed, attend, predict, append, repeat.

I ended with a question: **as the sequence grows, what do we need to keep in memory to avoid recomputing K and V for old tokens?**

That question drives this entire module. Today we measure exactly how bad the problem is. Next session, we fix it.

But first — let me show you the problem is real, not just theoretical.

---

## PART 1 — The Naive Decode Loop (15 min)

> **[Open notebook, Section: "Why naive decoding is expensive"]**

### What "Naive" Means (3 min)

When I say "naive decoding," I mean the most straightforward implementation of autoregressive generation. No tricks, no optimizations. Just:

1. Take the full sequence so far.
2. Run the entire model over the entire sequence.
3. Take the last position's output, pick the most likely next token.
4. Append it to the sequence.
5. Go back to step 1.

This is exactly what you built in Session 00. It's correct — the math is right, the attention is right, the causal mask is right. But it's doing something wasteful that you might not have noticed yet.

### The Waste (5 min)

> **[Step away from the notebook. Talk directly.]**

Think about what happens at decode step 3, when the sequence is `[token_0, token_1, token_2, token_3]`.

The model computes:
- K and V for token_0
- K and V for token_1
- K and V for token_2
- K and V for token_3

Then it uses these to compute attention and predict token_4.

Now step 4. The sequence is `[token_0, token_1, token_2, token_3, token_4]`. The model computes:
- K and V for token_0 — **again**
- K and V for token_1 — **again**
- K and V for token_2 — **again**
- K and V for token_3 — **again**
- K and V for token_4 — new

Four out of five K/V computations were wasted. Token_0's embedding didn't change. Token_0's Q, K, V didn't change. We just did the same math twice.

Step 5: sequence is 6 tokens. Five out of six K/V computations are wasted.

> **[Write on the board:]**

```
Step 1: 4 tokens  → 4 K/V computed → 0 reused
Step 2: 5 tokens  → 5 K/V computed → 4 reused (wasted)
Step 3: 6 tokens  → 6 K/V computed → 5 reused (wasted)
Step 4: 7 tokens  → 7 K/V computed → 6 reused (wasted)
Step 5: 8 tokens  → 8 K/V computed → 7 reused (wasted)
```

> **[Ask]**

How many total K/V computations happen across all 5 steps?

**Answer:** 4 + 5 + 6 + 7 + 8 = 30. But only 8 of those are for *new* tokens. The other 22 are recomputations.

That's the waste. Now let me show you it's not just a counting exercise — it has real cost.

---

## PART 2 — Live Demo: The Recomputation Count (15 min)

> **[Open notebook, Section: "Hand-trace: what gets recomputed each step"]**

### Setting Up (3 min)

> **[Run the first code cell — generate_naive demo]**

```python
torch.manual_seed(0)
vocab_size, d_model = 50, 32
embed = nn.Embedding(vocab_size, d_model)
lm_head = nn.Linear(d_model, vocab_size, bias=False)
attn = TinyCausalAttention(d_model)

prompt = torch.randint(0, vocab_size, (1, 4))
out = generate_naive(attn, prompt, embed, lm_head, max_new_tokens=8)
print("prompt length 4 + 8 new tokens ->", tuple(out.shape))
```

Output: `(1, 12)`. Four prompt tokens plus eight generated tokens. The model produced 8 new tokens, one at a time.

This is a tiny model — `d_model=32`, single attention head. On your laptop, this runs instantly. But that's deceptive. Let me show you what's happening inside.

### Instrumenting the Forward Pass (5 min)

> **[Run the CountingAttention code cell]**

Here's the trick. I've subclassed `TinyCausalAttention` with a `CountingAttention` version that logs how many tokens it processes on each call. The math is identical — I'm just counting.

```python
class CountingAttention(TinyCausalAttention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.tokens_processed_this_call = x.shape[1]
        return super().forward(x)
```

Now let's run the decode loop and watch:

> **[Run the loop code cell]**

```
naive decode: full recompute every step
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ decode step ┃ sequence length fed to model ┃ tokens recomputed (all of them)┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1           │                            4 │                               4 │
│ 2           │                            5 │                               5 │
│ 3           │                            6 │                               6 │
│ 4           │                            7 │                               7 │
│ 5           │                            8 │                               8 │
└─────────────┴──────────────────────────────┴───────────────────────────────┘
```

Look at column 3: **tokens recomputed**. At every step, every token in the sequence is recomputed. Step 1 processes 4 tokens. Step 2 processes 5. Step 3 processes 6.

The "tokens recomputed" column equals the "sequence length" column every time. That's the waste.

> **[Ask]**

What would this table look like with a cache?

**Answer:** The "tokens recomputed" column would be `1` for every row. Only the new token gets computed. The old tokens' K and V are pulled from memory, not recomputed.

Let me quantify how much faster that would be.

### The Math (7 min)

> **[Write on the board:]**

Total attention work across `n` generated tokens (no cache):

$$W = 1 + 2 + 3 + \dots + n = \frac{n(n+1)}{2}$$

This is **quadratic** in `n`. Doubling the sequence length quadruples the total work.

With a cache, each step only processes 1 new token:

$$W = 1 + 1 + 1 + \dots + 1 = n$$

This is **linear** in `n`. Doubling the sequence length doubles the total work.

> **[Write the comparison:]**

```
n = 8 tokens:    naive = 36 work units,  cached = 8 work units  → 4.5x
n = 32 tokens:   naive = 528,            cached = 32              → 16.5x
n = 128 tokens:  naive = 8,256,          cached = 128             → 64.5x
n = 256 tokens:  naive = 32,896,         cached = 256             → 128.5x
```

> **[Point at the numbers]**

The ratio between naive and cached is `n/2`. At 256 tokens, naive does 128 times more work than a cached version. At 1000 tokens, it's 500 times more. At 4096 tokens — a typical context length for modern models — it's over 2000 times more.

This isn't a small optimization. It's the difference between "works" and "doesn't work at scale."

---

## PART 3 — Wall-Clock Proof (15 min)

> **[Open notebook, Section: "Timing blowup across sequence length"]**

### Running the Benchmark (5 min)

> **[Run the timing code cell]**

```python
for n in [8, 32, 128, 256]:
    prompt = torch.randint(0, vocab_size, (1, 2))
    start = time.perf_counter()
    generate_naive(attn, prompt, embed, lm_head, max_new_tokens=n)
    elapsed = time.perf_counter() - start
```

```
naive decode wall-clock time vs. total generated length
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ total tokens generated ┃ time (s) ┃ n(n+1)/2 (relative work units)┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 8                      │   0.0006 │                             36 │
│ 32                     │   0.0017 │                            528 │
│ 128                    │   0.0113 │                           8256 │
│ 256                    │   0.0262 │                          32896 │
└────────────────────────┴──────────┴──────────────────────────────┘
```

Look at the pattern. From 8 to 32 tokens (4x more tokens), time goes from 0.0006s to 0.0017s — roughly 3x. But the work units went from 36 to 528 — nearly 15x. Why the discrepancy?

**Answer:** This is a tiny model. The overhead of Python loops and embedding lookups dominates at small scales. The quadratic growth is in the *attention* computation, which is a small fraction of the total time when `d_model=32`.

On a real model — say Llama 3 8B with `d_model=4096` and 32 attention heads — the attention computation dominates. That's when the quadratic blowup becomes the bottleneck, not a rounding error.

### Connecting Formula to Reality (5 min)

> **[Step away from the notebook]**

Let me make this concrete with a real model. Llama 3 8B:

- `d_model` = 4096
- `num_heads` = 32
- `head_dim` = 128
- `num_layers` = 32

At 4096 tokens of context, naive decoding recomputes attention for every token at every step. The total attention FLOPs across generation are proportional to `n² × d_model`. For `n = 4096`:

```
4096² × 4096 ≈ 68 billion FLOPs just for attention recomputation
```

With a cache, that drops to:

```
4096 × 4096 ≈ 16 million FLOPs
```

That's a 4000x reduction. The cache doesn't just make it faster — it makes generation *feasible* at scale.

### The FLOPs Estimate (5 min)

> **[Introduce the assignment's core function]**

The assignment for this session asks you to implement a FLOPs estimate. Here's the reasoning:

One self-attention forward pass over `seq_len` tokens:
- Q, K, V projections: `3 × seq_len × d_model × d_model` FLOPs (but this is constant — it doesn't depend on sequence length in the same way)
- The attention scores: `Q @ K.T` is a matrix multiply of `(seq_len, d_k) × (d_k, seq_len)` = `2 × seq_len × seq_len × d_k` FLOPs
- The attention output: `weights @ V` is another `(seq_len, seq_len) × (seq_len, d_k)` = same cost

The key term is `seq_len²`. That's the quadratic part. The FLOPs estimate for one attention pass is roughly:

```python
def count_flops_estimate(seq_len: int, d_model: int) -> int:
    return 2 * seq_len * seq_len * d_model
```

Doubling `seq_len` quadruples the FLOPs. That's the signature of quadratic growth.

---

## PART 4 — The Fix: KV Cache Preview (10 min)

> **[Open notebook, Section: Mermaid diagram + Recap]**

### What a Cache Would Do (5 min)

> **[Show the Mermaid sequence diagram]**

```
step 1: forward over tokens [1..4]  →  recompute K,V for 1,2,3,4
step 2: forward over tokens [1..5]  →  recompute K,V for 1,2,3,4,5 (1-4 redone!)
step 3: forward over tokens [1..6]  →  recompute K,V for 1,2,3,4,5,6 (1-5 redone!)
```

Every arrow in this diagram says "recompute." The fix is to replace "recompute" with "reuse."

Here's the idea:

1. **First time you see a token**, compute its K and V, and **store them** in a cache.
2. **Every subsequent step**, pull the cached K and V for all previous tokens. Only compute K and V for the **new token**.
3. **Append** the new K and V to the cache.
4. **Run attention** over the new Q against all cached K and V (old + new).

That's it. The cache is just a buffer of K and V tensors, one per layer, growing by one entry per generated token.

> **[Write on the board:]**

```
With cache, step 2:
  - Pull K,V for tokens 1,2,3,4 from cache (no computation)
  - Compute K,V for token 5 only (1 new computation)
  - Concatenate: K = [cached_K, new_K], V = [cached_V, new_V]
  - Run attention normally
```

### Why This Works (3 min)

The key insight is that **K and V for old tokens never change.** Token 1's embedding is fixed — it was set when we first processed it. The model's weights are frozen at inference time. So `K_1 = W_K @ embed_1` is the same value whether we compute it once or a thousand times.

Caching exploits this constancy. We compute once, store, and reuse.

This is not an approximation. The output is **mathematically identical** to naive decoding. Every session in this module that introduces a cache or a variant preserves this identity — the only thing that changes is how much memory and compute we use.

### The Catch (2 min)

> **[Plant the seed for Session 02]**

There's an obvious tradeoff. Naive decoding uses almost no memory — it processes the sequence and discards the intermediates. A cache **stores** those intermediates. For every token we've generated, we're keeping its K and V in GPU memory.

How much memory does that cost? That's exactly what Session 02 will calculate. The formula is:

```
KV_cache_bytes = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes × seq_len
```

For a 7B model at 4096 tokens of context, that's about **1 GB** of GPU memory just for the cache. At 128K tokens, it's **32 GB** — more than most GPUs have.

The cache solves the compute problem but creates a memory problem. That tension — compute vs. memory — is what drives the rest of this module.

---

## PART 5 — What This Session Proved (5 min)

> **[Step away from the notebook. Direct address.]**

Let me summarize what we just demonstrated.

**The claim:** Naive autoregressive decoding does O(n²) total work to generate n tokens.

**The proof:** We instrumented the forward pass and counted. At every decode step, the model recomputes K and V for the *entire* sequence — including tokens that haven't changed. We showed this with a counting table, a wall-clock benchmark, and the `n(n+1)/2` formula.

**The consequence:** On a real model, naive decoding at 4096 tokens of context does 4000x more attention work than necessary. Generation is not just slow — it's quadratically slow.

**The fix:** Cache K and V. Compute once, reuse forever. This turns O(n²) into O(n).

**The cost:** The cache consumes GPU memory proportional to sequence length. How much memory, exactly? Session 02.

---

## PART 6 — Assignment Walkthrough (5 min)

> **[Open `assignments/01_naive_decoding/task.md`]**

Your assignment for this session has two parts.

**Part 1: Implement `count_flops_estimate`**

```python
def count_flops_estimate(seq_len: int, d_model: int) -> int:
    """Rough FLOPs estimate for one full self-attention forward pass."""
    return 2 * seq_len * seq_len * d_model
```

The test verifies that going from `seq_len=10` to `seq_len=100` (10x) produces more than 10x the FLOPs. If your function returns a linear relationship, it's wrong — the answer must be quadratic.

**Part 2: Explain in one sentence**

The acceptance criteria say: "You can explain, in one sentence, why per-step latency grows with position."

Here's a model answer: **"Each new token recomputes attention over the entire sequence so far, so per-step work grows linearly with position, making total work quadratic."**

> **[Ask]**

Can someone rephrase that in their own words?

Wait for an answer. Then: "Good. That's the entire motivation for this module. Next session, we calculate exactly how much memory the fix costs."

---

## CLOSING — Bridge to Session 02 (5 min)

> **[Direct address]**

We've now completed the first two links in the chain:

1. **Session 0:** Here's how attention works.
2. **Session 1:** Here's why running it naively is wasteful.

The next two links:

3. **Session 02:** Here's how much memory the fix costs. We'll derive the KV cache memory formula and compute real numbers for real model configs. You'll see why a 7B model at 128K context needs 32 GB just for the cache.

4. **Session 03:** Here's the MHA baseline — the reference architecture that all variants compare against.

By the end of Session 03, you'll have a working cached attention module. Sessions 04–07 will then ask: how do we make that cache smaller without losing quality?

> **[Final line]**

One sentence to remember: **naive decoding recomputes what it should be reusing.** That's the problem. Everything else is the solution.

---

## APPENDIX — Timing Summary

| Part | Topic | Minutes |
|------|-------|---------|
| 0 | Recap and frame | 5 |
| 1 | The naive decode loop | 15 |
| 2 | Live demo: recomputation count | 15 |
| 3 | Wall-clock proof | 15 |
| 4 | KV cache preview | 10 |
| 5 | What this session proved | 5 |
| 6 | Assignment walkthrough | 5 |
| — | Closing / bridge to Session 02 | 5 |
| **Total** | | **~75 min** |

## APPENDIX — Key Moments to Pause

1. **After Part 1, the board exercise:** "How many total K/V computations across 5 steps?" — makes them feel the waste before quantifying it.
2. **After Part 2, the ratio question:** "What would the table look like with a cache?" — primes them for the fix.
3. **After Part 3, the real-model calculation:** "68 billion FLOPs vs 16 million" — makes the quadratic cost visceral.
4. **After Part 4, the catch:** "The cache solves compute but creates memory" — sets up Session 02.

## APPENDIX — Key Terms Introduced

| Term | Definition | Where it appears |
|------|-----------|-----------------|
| Naive decoding | Autoregressive generation with full recomputation at every step | Part 1 |
| Recompute | Computing K/V for tokens whose embeddings haven't changed | Part 1 |
| Quadratic growth | O(n²) total work — doubling n quadruples work | Part 2 |
| KV cache | Buffer storing previously computed K/V tensors for reuse | Part 4 |
| Per-step cost | Amount of work done at each decode step | Part 2 |
| Wall-clock time | Actual elapsed time, not just theoretical FLOPs | Part 3 |
