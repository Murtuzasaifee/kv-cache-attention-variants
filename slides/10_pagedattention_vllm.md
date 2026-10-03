# Session 10 — PagedAttention & vLLM

- Problem: contiguous KV cache allocation wastes memory (over-provisioning for worst-case length, fragmentation across requests)
- PagedAttention (virtual-memory-inspired): KV cache split into fixed-size blocks, non-contiguous, tracked via a per-sequence block table
- Enables near-zero-waste memory sharing (e.g. prefix caching, beam search) and much higher concurrent-request throughput
- SCOPE NOTE: this session is install-and-benchmark, not a from-scratch reimplementation of block-table paging
- Local repo: benchmark harness + expected-output tables only. Actual vLLM install/run: `colab/session10_vllm_pagedattention_colab.ipynb` (GPU runtime)
- Compare: vLLM concurrent throughput vs naive HF `generate()` loop, using the SAME `bench.benchmark` harness from Sessions 8/9
