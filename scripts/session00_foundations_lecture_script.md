# Session 00 — Foundations: Lecture Script

**Duration:** ~90 minutes
**Materials:** Jupyter notebook `teaching_notebooks/00_foundations.ipynb`, projector
**Prerequisite:** None. This is where the course starts.

---

## PART 0 — The Bigger Picture (7 min)

> **[0:00 — Standing, before any slides]**

Let me tell you what this module is about before we write a single line of code.

We're going to spend 10 sessions on one question: **when a large language model generates text, token by token, what does it need to remember about the tokens it has already seen — and what does it cost to remember it?**

That's it. Everything in this module — KV cache, MQA, GQA, MLA, RoPE, FlashAttention, PagedAttention — is a different answer to that one question, motivated by a different failure.

Here's how it works. Right now, you all have a working transformer from the previous module. You know how self-attention works, you know what masking is, you've built a decoder-only model. This module takes that working model and asks: **what happens when you try to run it in production?**

And the answer is: it falls over. Not because the math is wrong, but because the engineering doesn't scale.

The story of this module is a sequence of failures and fixes:

- **Session 1:** Your model works, but generation gets slower with every token. Why?
- **Session 2:** You fix the speed, but now you're running out of GPU memory. How much memory, exactly?
- **Session 3–7:** You reduce the memory, but each reduction trades off against model quality. How do you choose?
- **Session 8–9:** Meanwhile, the attention *computation itself* is slow — not just the memory. There's a faster way that gives identical results.
- **Session 10:** You've optimized for one request. What happens when a thousand users hit your API at the same time?

Every session introduces a problem, and the next session fixes it. By the end, you'll be able to look at any modern LLM — Llama 3, Qwen 3, DeepSeek, Gemma — and explain *why* it uses the attention variant it uses, and how much memory its KV cache costs.

Today — Session 0 — we go back to first principles. No caching, no optimization. Just: how does a transformer process language, one token at a time?

> **[Pause. Let that land.]**

Let's start with the most basic question: how does a language model even see text?

---

## PART 1 — Tokens and Embeddings (12 min)

> **[Open notebook, Section 1]**

### The Setup (3 min)

A language model never sees raw text. It doesn't know what "cat" means. What it sees is **integers** — token IDs that a tokenizer assigns to sub-words.

Think of it this way. Before you can do math on words, you need a numbering system. The tokenizer gives you that. "The" becomes 0. "Cat" becomes 1. "Sat" becomes 2. Now you have numbers you can feed into a neural network.

But integers alone aren't useful. The number 1 doesn't tell you anything about what "cat" means. So the model has a second step: **embedding lookup**.

The embedding table is a big matrix — one row per token in the vocabulary. Row `i` is a learned vector that represents token `i`. The model has learned, through training, to put tokens that appear in similar contexts near each other in this vector space.

**Analogy:** Imagine a huge map of the English language. Every word is a point on this map. "Cat" and "dog" are close together because they appear in similar sentences. "Cat" and "democracy" are far apart. The embedding table *is* this map — except it's not 2D, it's hundreds of dimensions.

### The Code (5 min)

> **[Run the code cell]**

```python
sentence = ["the", "cat", "sat"]
token_ids = torch.tensor([vocab.index(w) for w in sentence])
embeddings = embedding_table[token_ids]
print("embeddings shape:", tuple(embeddings.shape))
```

Look at the output:

```
sentence: ['the', 'cat', 'sat']
token ids: [0, 1, 2]
embeddings shape: (3, 8)
```

Three things just happened:

1. **Tokenization:** Three words became three integers: `[0, 1, 2]`.
2. **Lookup:** We pulled three rows out of the embedding table — row 0, row 1, row 2.
3. **Result:** A `(3, 8)` tensor. Three tokens, each represented by an 8-dimensional vector.

This `(seq_len, d_model)` shape is going to appear everywhere in this course. When we say `d_model`, we mean the size of these vectors — in real models, it's thousands. Here it's 8, so we can see every number.

### Checkpoint Question (4 min)

> **[Pause and ask]**

If I gave you a sentence of 50 tokens, what would the shape of the embeddings tensor be?

**Answer:** `(50, 8)` — 50 rows, each an 8-dimensional vector. The sequence length is the first dimension. This is important because **everything that happens next operates on this sequence.** The longer the sequence, the more work attention has to do.

Let me show you what that work looks like.

---

## PART 2 — Self-Attention Intuition (10 min)

> **[Open notebook, Section 2]**

### The Analogy (5 min)

Every token in our sequence produces three vectors from its embedding:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer, as a label?"
- **Value (V)** — "what do I actually contribute if picked?"

Here's the analogy that makes this stick.

Imagine a library. You walk in with a search query — "I need books about ancient Rome." That's your **Query**.

Every book on the shelf has a catalogued set of subject tags — "history," "politics," "Mediterranean." Those are the **Keys**.

You type your query into the search system. It compares your query against every book's tags and returns a ranked list. Books whose tags match your query get ranked higher. That ranking is the **attention weight**.

What you get back is not the books themselves — it's a **weighted mix** of their contents. If book A matched your query strongly and book B matched weakly, you get mostly A's content and a little of B's. The content itself — the actual text you read — is the **Value**.

The key insight: **every token does this simultaneously.** Token 0 produces a query, compares it against every other token's key, and collects a weighted mix of their values. Token 1 does the same. Token 2 does the same. They all run in parallel.

This is why it's called **self-attention** — the sequence is attending to itself. Each token asks "which other tokens in this sequence are relevant to me?" and gets a weighted answer.

### The Three Vectors (5 min)

Let me make this concrete with our three-token example.

Token "the" has an embedding. From that embedding, the model computes:
- A query: "I'm looking for nouns that follow articles."
- A key: "I'm the word 'the' — I'm a function word."
- A value: "My contribution is grammatical structure context."

Token "cat" has an embedding. From that embedding:
- A query: "I'm looking for what precedes me and what follows me."
- A key: "I'm a noun, an animal."
- A value: "My contribution is semantic content."

Token "sat" has an embedding:
- A query: "I'm looking for my subject."
- A key: "I'm a verb, past tense."
- A value: "My contribution is action context."

When we compute attention, "sat"'s query matches "cat"'s key strongly — "sat" is looking for a subject, "cat" is a noun. So "cat"'s value contributes heavily to "sat"'s output. "The" contributes less to "sat" because articles aren't strong subjects.

This is how the model learns relationships between tokens — not through rules, but through learned query-key similarity.

---

## PART 3 — Scaled Dot-Product Attention (20 min)

> **[Open notebook, Section 3]**

### The Formula (5 min)

Now let's see the math. This is the core equation of attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Let me break this down piece by piece:

1. **QK^T** — matrix multiply queries against keys. This gives you a `(seq_len × seq_len)` matrix of raw scores. Each entry `(i, j)` is "how much does token `i` want to attend to token `j`?"

2. **Divide by sqrt(d_k)** — scale the scores down. Why? Because if `d_k` is large (say 128), the dot products grow with dimension. Large dot products make softmax output very peaked — almost all weight on one token, almost zero on everything else. Dividing by `sqrt(d_k)` keeps the dot products in a reasonable range so softmax produces smooth, useful weights.

3. **Softmax** — normalize each row so the weights sum to 1. Now you have a probability distribution: "token `i` allocates X% of its attention to token `j`."

4. **Multiply by V** — the weighted sum. Each token's output is a mix of all values, weighted by the attention scores.

### By Hand First (5 min)

> **[Run the code cell — this is critical, walk through every line]**

```python
d_k = 4
Q = torch.randn(3, d_k)
K = torch.randn(3, d_k)
V = torch.randn(3, d_k)

scores = Q @ K.T / math.sqrt(d_k)
```

Three tokens, head dimension 4. We compute `Q @ K.T` — that's a `(3, 3)` matrix. One row per query token, one column per key token.

Look at the raw scores:

```
raw scores (3x3, one row per query token):
 tensor([[-0.2284,  0.5448, -0.3085],
        [ 0.8640,  0.4951,  0.3000],
        [-0.6569, -0.0357, -0.4154]])
```

Row 0: token 0's query scored 0.54 against token 1's key — that's the highest in the row. So token 0 wants to attend most to token 1.

Row 1: token 1's query scored 0.86 against token 0's key — highest in its row. Token 1 attends most to token 0.

Row 2: token 2's query scored −0.04 against token 1's key — moderate. It's the least confident of the three.

Now softmax normalizes each row:

```
attention weights (each row sums to 1):
 tensor([[0.2445, 0.5298, 0.2257],
        [0.4424,  0.3059,  0.2517],
        [0.2419,  0.4502,  0.3080]])
```

Row sums to exactly 1 — guaranteed by softmax. Token 0 puts 53% of its attention on token 1, 24% on token 0 itself, 23% on token 2.

Finally, the output is `weights @ V`:

```
output (3 tokens x d_k):
 tensor([[ 0.7457,  1.3025,  0.4068, -0.6725],
        [ 0.8677,  0.6612,  0.4144, -0.6146],
        [ 0.5289,  1.0287,  0.5016, -0.5933]])
```

Token 0's output is a blend: 24% of its own value, 53% of token 1's value, 23% of token 2's value. That's attention in action.

### The Shape Story (5 min)

> **[Draw this on the board or in a text editor]**

Let's track shapes through the whole computation:

```
Q: (3, 4)    — 3 tokens, 4 dims
K: (3, 4)    — 3 tokens, 4 dims
V: (3, 4)    — 3 tokens, 4 dims

Q @ K.T: (3, 3) — scores matrix: which token attends to which
softmax: (3, 3) — weights matrix: normalized attention

weights @ V: (3, 4) — output: one vector per token
```

The output has the same shape as the input embeddings. That's deliberate — each token gets a new representation that incorporates information from the tokens it attended to. This output gets passed to the next part of the transformer.

**Critical observation:** that `(3, 3)` scores matrix is the bottleneck. If you have 1000 tokens, it's `(1000, 1000)`. If you have 100,000 tokens, it's `(100000, 100000)`. This quadratic cost is going to matter later.

### Checkpoint Question (5 min)

> **[Pause and ask]**

Why do we divide by `sqrt(d_k)` instead of just `d_k`?

**Answer:** Because the variance of a dot product of two random vectors with `d_k` dimensions is proportional to `d_k`. Dividing by `sqrt(d_k)` normalizes the variance to 1 regardless of dimension. If we divided by `d_k`, the scores would shrink too much and softmax would produce nearly uniform weights — every token would contribute equally, which defeats the purpose.

---

## PART 4 — Causal Masking (12 min)

> **[Open notebook, Section 4]**

### Why Masking Exists (3 min)

There's a problem with what we just did.

In the attention we computed, token 2 could attend to token 3, token 4, token 5 — any token in the sequence. But during **generation**, the model produces tokens one at a time. When it's generating token 3, it should not be able to see token 4 — token 4 doesn't exist yet.

This is cheating. If the model could peek at future tokens, it would get the answer for free. It needs to figure out each token using only the tokens that came before it.

Causal masking solves this. Before softmax, we set all "future" scores to negative infinity. Softmax turns those into exactly zero. The model literally cannot see ahead.

### The Mask (3 min)

> **[Run the Rich table cell]**

Look at this table:

```
query pos \ key pos |  0 |  1 |  2 |  3 |  4 |
--------------------|----|----|----|----|----|
       0            |  . |  X |  X |  X |  X |
       1            |  . |  . |  X |  X |  X |
       2            |  . |  . |  . |  X |  X |
       3            |  . |  . |  . |  . |  X |
       4            |  . |  . |  . |  . |  . |
```

`X` means blocked. `.` means allowed.

Row 0: token 0 can only see itself. It's the first token — nothing came before it.
Row 1: token 1 can see token 0 and itself.
Row 4: token 4 can see all five tokens — everything came before it.

This is an upper-triangular mask. It's the defining structure of **causal** (or "autoregressive") attention.

### The Proof (6 min)

> **[Run the masking code cell]**

```python
causal_scores = torch.randn(seq_len, seq_len).masked_fill(mask, float("-inf"))
causal_weights = torch.softmax(causal_scores, dim=-1)
```

Let's look at what happens:

```
row 0 (only sees itself): tensor([1., 0., 0., 0., 0.])
row 4 (sees all 5 tokens): tensor([0.079, 0.218, 0.186, 0.234, 0.283])
```

Row 0: weight is `[1, 0, 0, 0, 0]`. Token 0 is forced to attend only to itself. It has no choice — all other positions are −∞, which softmax turns to 0.

Row 4: weight is distributed across all 5 positions. Token 4 can see everyone.

**This is the key constraint of autoregressive generation:** each token can only attend to tokens at positions ≤ its own position. No peeking ahead.

### Connection to Generation (3 min)

> **[Step back from the code]**

Think about what this means for generation. When the model generates token 5:

1. It runs attention over tokens 0–5.
2. The causal mask ensures token 5's query only interacts with keys from tokens 0–4 (and itself).
3. It cannot use any information from token 6, 7, 8 — they don't exist yet.

This is correct and necessary. But notice: **every time we generate a new token, we recompute attention over all previous tokens.** Token 5 recomputes against 0–4. Token 6 recomputs against 0–5. Token 7 recomputes against 0–6.

That's a lot of redundant work. We'll quantify exactly how much in Session 1.

---

## PART 5 — Autoregressive Generation (15 min)

> **[Open notebook, Section 5]**

### The Loop (5 min)

> **[Run the generation code cell]**

Here's the full generation loop:

```python
prompt = torch.randint(0, vocab_size, (1, 3))
ids = prompt
for step in range(4):
    x = embed(ids)
    hidden = toy_transform(x)
    next_id = lm_head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=1)
```

Step by step:

1. Start with a 3-token prompt: `[42, 27, 38]`
2. Embed all tokens: `(1, 3, 16)` — batch of 1, sequence of 3, d_model of 16
3. Run through the model (here it's identity, but in a real transformer it's self-attention + feed-forward layers)
4. Take the **last position's** output — that's the prediction for the next token
5. Argmax to get the most likely next token
6. Append it to the sequence
7. Repeat

Output:

```
step 0: generated token 25, sequence so far: [42, 27, 38, 25]
step 1: generated token 33, sequence so far: [42, 27, 38, 25, 33]
step 2: generated token 9,  sequence so far: [42, 27, 38, 25, 33, 9]
step 3: generated token 6,  sequence so far: [42, 27, 38, 25, 33, 9, 6]
```

Each step, the sequence grows by one. Each step, the model processes the entire sequence from scratch.

### The Problem Hidden in This Loop (10 min)

> **[This is the most important moment of the session. Slow down.]**

Look at step 0. The model processes tokens `[42, 27, 38]`. It computes attention across all 3 tokens. That's fine.

Now look at step 1. The model processes tokens `[42, 27, 38, 25]`. It computes attention across all 4 tokens.

**But token 42, 27, and 38 are the same as they were in step 0.** Their embeddings haven't changed. Their K and V vectors haven't changed. We're recomputing them from scratch.

Step 2: `[42, 27, 38, 25, 33]`. Now we recompute attention over 5 tokens. Tokens 42, 27, 38, 25 are recomputed *again.*

Step 3: `[42, 27, 38, 25, 33, 9]`. Six tokens. Four of them are recomputed for the third time.

> **[Write this on the board:]**

```
Step 0: compute attention for 3 tokens
Step 1: compute attention for 4 tokens (3 are recomputed)
Step 2: compute attention for 5 tokens (4 are recomputed)
Step 3: compute attention for 6 tokens (5 are recomputed)
```

> **[Ask the class]**

If I generate 50 tokens, how many times does token 0 get recomputed?

**Answer:** 49 times. Once in each step after the first.

If I generate 1000 tokens, how many total attention computations happen?

**Answer:** 1 + 2 + 3 + ... + 999 ≈ 500,000. That's quadratic growth. Double the sequence, quadruple the work.

This is the **exact problem** that motivates everything in this module. The model is doing redundant work — recomputing K and V for tokens it has already seen. Session 1 will measure this empirically. Session 2 will quantify the memory cost. Sessions 3–7 will propose architectural solutions.

> **[Let this sink in for a moment. This is the hook.]**

---

## PART 6 — Transformer Decoder Block (8 min)

> **[Open notebook, Section 6]**

### Where Attention Lives (3 min)

We've been talking about attention as if it's the whole model. It's not. Here's what a single decoder layer actually looks like:

```
x → self-attention → add & norm → feed-forward → add & norm → output
```

Each layer has two sub-blocks:
1. **Self-attention** — what we just learned. This is where tokens exchange information.
2. **Feed-forward** — a small neural network applied independently to each token. This is where each token processes the information it gathered from attention.

There's also **add & norm** (residual connection + layer normalization) between them, which stabilizes training.

> **[Show the Mermaid data flow diagram]**

This is the flow of data through one layer:

```
token ids → embedding lookup → Q, K, V projections → scores → causal mask → softmax → weighted sum of V → output projection
```

Every session in this module focuses on one part of this pipeline. The **Q, K, V projections** are where MQA, GQA, and MLA change things. The **scores** computation is where FlashAttention changes things. The **output** is where the cache lives.

The **feed-forward** part never changes. We don't touch it. The entire course is about optimizing the self-attention sub-block.

### The Sequence Diagram (5 min)

> **[Show the Mermaid sequence diagram]**

```
Query pos 0 → Key pos 0: allowed
Query pos 0 → Key pos 4: blocked (future)
Query pos 4 → Key pos 0: allowed
Query pos 4 → Key pos 4: allowed (self)
```

This is the causal mask visualized as a conversation. Token 0 can only talk to itself — it's the first word, there's nothing before it. Token 4 can talk to everyone — they all came before it.

The mask creates an asymmetry: **later tokens have more context.** Token 4 has access to 5 tokens' worth of information. Token 0 has access to 1. This is why the first few tokens of a generated sequence are often lower quality — they have less context to work with.

---

## CLOSING — The Question That Drives Everything (5 min)

> **[Step away from the notebook. Talk to the class directly.]**

Let me recap what we've built today:

1. **Tokens and embeddings:** Language becomes numbers, numbers become vectors.
2. **Self-attention:** Each vector produces Q, K, V. Attention is a weighted lookup — queries match keys, values are the result.
3. **Scaled dot-product:** The actual math. `softmax(QK^T / sqrt(d_k)) V`. Scores matrix is `(seq_len × seq_len)`.
4. **Causal masking:** We block future positions so the model can't cheat during generation.
5. **Generation loop:** Embed → attend → predict → append → repeat. The sequence grows each step.

And here's the problem you should be thinking about as you leave:

**As the sequence grows, every new token recomputes attention over all previous tokens. The K and V vectors for old tokens are identical every time — they never change. But we recompute them anyway.**

That's waste. Session 1 will measure exactly how much waste. Session 2 will show you how to eliminate it — by caching K and V instead of recomputing them.

But caching creates a new problem: memory. And that problem is what drives the rest of this module.

> **[If time permits, 2 min]**

One more thing. Everything we've seen today used a single attention head. Real models use multiple heads — they compute attention multiple times in parallel with different learned projections. We'll cover that in Session 3, when we recap MHA specifically for inference.

For now, the foundation is set. You understand the mechanism. Next session, we break it.

---

## APPENDIX — Timing Summary

| Part | Topic | Minutes |
|------|-------|---------|
| 0 | Bigger picture | 7 |
| 1 | Tokens & embeddings | 12 |
| 2 | Self-attention intuition | 10 |
| 3 | Scaled dot-product attention | 20 |
| 4 | Causal masking | 12 |
| 5 | Autoregressive generation | 15 |
| 6 | Transformer decoder block | 8 |
| — | Closing | 5 |
| **Total** | | **~89 min** |

## APPENDIX — Key Moments to Pause

1. **After Part 1, checkpoint:** "If I gave you a sentence of 50 tokens, what shape?"
2. **After Part 3, checkpoint:** "Why sqrt(d_k)?"
3. **After Part 5, the big moment:** "If I generate 50 tokens, how many times is token 0 recomputed?"
4. **Closing:** "What should we keep in memory?" — leave them with this question.
