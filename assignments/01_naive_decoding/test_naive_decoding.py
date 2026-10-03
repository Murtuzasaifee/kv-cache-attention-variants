import sys

sys.path.insert(0, ".")
from starter import count_flops_estimate


def test_flops_grow_quadratically_with_seq_len():
    small = count_flops_estimate(seq_len=10, d_model=32)
    large = count_flops_estimate(seq_len=100, d_model=32)
    # 10x longer sequence -> attention FLOPs (~seq_len^2 term) grow > 10x, not linearly
    assert large > small * 10
