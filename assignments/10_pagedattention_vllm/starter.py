"""Assignment 10 answer key — throughput ratio helper."""

import sys

sys.path.insert(0, "../../src")


def throughput_speedup(naive_mean_s: float, vllm_mean_s: float) -> float:
    return naive_mean_s / vllm_mean_s
