"""Session 9: unified attention via F.scaled_dot_product_attention (SDPA).

Bridge point of the module: from here on, hand-rolled attention math (mha.py/
gqa.py/mla.py) is replaced by PyTorch's backend-dispatching SDPA op. Backends:
math (always available, any device), efficient/flash (CUDA, and flash as of
2.13 also dispatches on Apple Silicon MPS for supported shapes), cudnn (CUDA).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.getLogger(__name__)

BACKEND_NAMES = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def run_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    backend: str = "math",
    is_causal: bool = True,
) -> Tensor:
    """Run SDPA forcing a specific backend, falling back to math on failure.

    Forcing flash/efficient/cudnn on a device or dtype/shape combo they don't
    support raises RuntimeError — that failure IS the teaching point (backend
    availability is hardware/shape dependent), so we log it and fall back
    rather than hide it.
    """
    if backend not in BACKEND_NAMES:
        raise ValueError(f"unknown backend {backend!r}, choose from {list(BACKEND_NAMES)}")

    try:
        with sdpa_kernel(BACKEND_NAMES[backend]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    except RuntimeError as e:
        logger.warning("backend=%s unavailable here (%s); falling back to math", backend, e)
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def available_backends() -> list[str]:
    """Backends that actually run on the current device without falling back."""
    q = k = v = torch.randn(1, 2, 4, 8)
    working = []
    for name, backend in BACKEND_NAMES.items():
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            working.append(name)
        except RuntimeError:
            continue
    return working


def default_device() -> torch.device:
    """Mac-first device pick: MPS if present, else CPU. CUDA is opt-in via --device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _demo() -> None:
    q = k = v = torch.randn(1, 2, 4, 8)
    out = run_sdpa(q, k, v, backend="math")
    assert out.shape == q.shape
    backends = available_backends()
    assert "math" in backends
    print(f"device={default_device()}, working backends: {backends}")


if __name__ == "__main__":
    _demo()
