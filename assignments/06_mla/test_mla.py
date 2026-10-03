import sys

sys.path.insert(0, ".")
from starter import bytes_per_token, mla_bytes_per_token


def test_mla_smaller_than_mha_at_matched_heads():
    mha_bpt = bytes_per_token(num_heads=8, head_dim=64, num_kv_heads=8)
    mla_bpt = mla_bytes_per_token(latent_dim=64, rope_dim=16)
    assert mla_bpt < mha_bpt
