"""Assignment 08 answer key — backend benchmarking."""

import sys

sys.path.insert(0, "../../src")

from kv_cache_variants.sdpa_backends import available_backends, run_sdpa
from kv_cache_variants.bench import benchmark


def fastest_backend(q, k, v) -> str:
    results = {b: benchmark(run_sdpa, q, k, v, backend=b)["mean_s"] for b in available_backends()}
    return min(results, key=results.get)
