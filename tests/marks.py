import pytest
import torch

__all__ = ["skip_if_cuda_not_available"]

skip_if_cuda_not_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available."
)
