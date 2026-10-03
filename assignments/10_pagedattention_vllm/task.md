# Assignment 10 — vLLM Throughput Benchmark (Colab)

## Objective
Run `colab/session10_vllm_pagedattention_colab.ipynb` on a GPU runtime,
install vLLM, and benchmark concurrent-request throughput against a naive
HF `generate()` loop using `bench.benchmark`.

## Starter
See `starter.py` for the throughput-ratio helper used to summarize results
locally (works without a GPU — just needs two mean_s numbers from the Colab run).

## Acceptance Criteria
- `test_throughput_ratio.py` passes.
