import sys

sys.path.insert(0, ".")
from starter import required_scale_factor


def test_scale_factor():
    assert required_scale_factor(4096, 8192) == 2.0
    assert required_scale_factor(4096, 4096) == 1.0
