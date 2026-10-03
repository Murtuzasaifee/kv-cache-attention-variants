import sys

sys.path.insert(0, ".")
from starter import mqa_savings_factor


def test_savings_factor_equals_num_heads():
    factor = mqa_savings_factor(num_heads=8, num_layers=4, head_dim=64, seq_len=1000)
    assert factor == 8.0
