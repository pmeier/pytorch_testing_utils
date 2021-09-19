import warnings
from typing import Any, Callable, Optional, Union, cast

import numpy as np
import pytest
import torch
from _pytest.python_api import ApproxBase, ApproxNumpy

from .attributes import assert_tensor_attributes_equal

__all__ = [
    "approx",
    "assert_tensor_equal",
    "assert_tensor_allclose",
]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return cast(np.ndarray, x.detach().cpu().numpy())


class ApproxPytorch(ApproxNumpy):
    def __init__(self, expected: torch.Tensor, **kwargs: Any):
        super().__init__(self._cast(expected), **kwargs)

    def __eq__(self, actual: Any) -> bool:
        if isinstance(actual, torch.Tensor):
            actual = self._cast(actual)
        return super().__eq__(actual)

    def _cast(self, x: torch.Tensor) -> Union[float, np.ndarray]:
        is_scalar = x.dim() == 0
        if is_scalar:
            return x.detach().cpu().item()
        else:
            return _to_numpy(x)


def approx(
    expected: Any,
    rel: Optional[float] = None,
    abs: Optional[float] = None,
    nan_ok: bool = False,
) -> ApproxBase:
    approximator = (
        ApproxPytorch if isinstance(expected, torch.Tensor) else pytest.approx
    )
    return approximator(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def assert_tensor_equal(
    actual: torch.Tensor,
    desired: torch.Tensor,
    attributes: bool = True,
    warn_floating_point: bool = True,
    verbose: bool = True,
    msg: Optional[str] = None,
    **attribute_kwargs: bool,
) -> None:
    if attributes:
        assert_tensor_attributes_equal(actual, desired, msg=msg, **attribute_kwargs)

    is_floating_point = (
        actual.dtype.is_floating_point or desired.dtype.is_floating_point
    )
    if is_floating_point and warn_floating_point:
        msg = (
            "Due to the limitations of floating point arithmetic, comparing "
            "floating-point tensors for equality is not recommended. Use "
            "assert_allclose instead."
        )
        warnings.warn(msg, RuntimeWarning)

    err_msg = msg if msg is not None else ""
    np.testing.assert_equal(
        _to_numpy(actual), _to_numpy(desired), verbose=verbose, err_msg=err_msg
    )


def _tensor_equal_asserter(
    **tensor_equal_kwargs: Any,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    return lambda actual, desired: assert_tensor_equal(
        actual, desired, **tensor_equal_kwargs,
    )


def assert_tensor_allclose(
    actual: torch.Tensor,
    desired: torch.Tensor,
    attributes: bool = True,
    rtol: float = 1e-7,
    atol: float = 0.0,
    equal_nan: bool = True,
    verbose: bool = True,
    msg: Optional[str] = None,
    **attribute_kwargs: bool,
) -> None:
    if attributes:
        assert_tensor_attributes_equal(actual, desired, msg=msg, **attribute_kwargs)

    err_msg = msg if msg is not None else ""
    np.testing.assert_allclose(
        _to_numpy(actual),
        _to_numpy(desired),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        verbose=verbose,
        err_msg=err_msg,
    )


def _tensor_allclose_asserter(
    **tensor_allclose_kwargs: Any,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    return lambda actual, desired: assert_tensor_allclose(
        actual, desired, **tensor_allclose_kwargs
    )
