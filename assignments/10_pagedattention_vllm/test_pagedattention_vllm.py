import sys

sys.path.insert(0, ".")
from starter import throughput_speedup


def test_speedup_ratio():
    assert throughput_speedup(naive_mean_s=2.0, vllm_mean_s=0.5) == 4.0
