"""Assignment 07 answer key — RoPE scale factor."""

import sys

sys.path.insert(0, "../../src")


def required_scale_factor(trained_max_len: int, target_len: int) -> float:
    return target_len / trained_max_len
