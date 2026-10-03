"""Sessions 8/9 (and reusable for Session 10): generic latency/throughput harness.

Written once, reused unmodified across sessions — don't rewrite per session.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    iters: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    """Time `fn(*args, **kwargs)` over `iters` runs after `warmup` untimed runs.

    Synchronizes CUDA/MPS around each timed call so GPU-async dispatch doesn't
    make timings look artificially fast.

    Returns dict with mean_s, min_s, max_s, throughput_per_s (1 / mean_s).
    """

    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()

    for _ in range(warmup):
        fn(*args, **kwargs)
    _sync()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(*args, **kwargs)
        _sync()
        times.append(time.perf_counter() - start)

    mean_s = sum(times) / len(times)
    return {
        "mean_s": mean_s,
        "min_s": min(times),
        "max_s": max(times),
        "throughput_per_s": 1.0 / mean_s if mean_s > 0 else float("inf"),
    }


def _demo() -> None:
    def slow_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    x = torch.randn(1000, 1000)
    result = benchmark(slow_add, x, x, warmup=2, iters=5)
    assert result["mean_s"] > 0
    assert set(result) == {"mean_s", "min_s", "max_s", "throughput_per_s"}
    print(f"benchmark result: {result}")


if __name__ == "__main__":
    _demo()
