import warnings
from typing import Any, Optional, cast

import numpy as np
import pytest
import torch
from _pytest.python_api import ApproxBase

__all__ = [
    "assert_tensor_size_equal",
    "assert_tensor_dtype_equal",
    "assert_tensor_layout_equal",
    "assert_tensor_device_equal",
    # "assert_tensor_memory_format_equal",
    "assert_tensor_meta_equal",
    "assert_tensor_equal",
    "assert_tensor_allclose",
    "approx",
]


def _assert_equal(
    actual: Any, desired: Any, msg: Optional[str] = None, prefix: Optional[str] = None
) -> None:
    if msg is None:
        msg = f"{actual} != {desired}"
        if prefix is not None:
            msg = prefix + msg

    assert actual == desired, msg


def assert_tensor_size_equal(
    actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
) -> None:
    _assert_equal(
        tuple(actual.size()),
        tuple(desired.size()),
        msg=msg,
        prefix="Tensor size mismatch: ",
    )


def assert_tensor_dtype_equal(
    actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
) -> None:
    _assert_equal(
        actual.dtype, desired.dtype, msg=msg, prefix="Tensor dtype mismatch: "
    )


def assert_tensor_layout_equal(
    actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
) -> None:
    _assert_equal(
        actual.layout, desired.layout, msg=msg, prefix="Tensor layout mismatch: "
    )


def assert_tensor_device_equal(
    actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
) -> None:
    _assert_equal(
        actual.device, desired.device, msg=msg, prefix="Tensor device mismatch: "
    )


# FIXME: Tensor.memory_format is not a valid attribute.
# def assert_tensor_memory_format_equal(
#     actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
# ) -> None:
#     _assert_equal(
#         actual.memory_format,
#         desired.memory_format,
#         msg=msg,
#         prefix="Tensor memory format mismatch: ",
#     )


def assert_tensor_meta_equal(
    actual: torch.Tensor,
    desired: torch.Tensor,
    assert_size_equal: bool = True,
    assert_dtype_equal: bool = True,
    assert_device_equal: bool = True,
    assert_layout_equal: bool = True,
    msg: Optional[str] = None,
) -> None:
    if assert_size_equal:
        assert_tensor_size_equal(actual, desired, msg=msg)
    if assert_dtype_equal:
        assert_tensor_dtype_equal(actual, desired, msg=msg)
    if assert_device_equal:
        assert_tensor_device_equal(actual, desired, msg=msg)
    if assert_layout_equal:
        assert_tensor_layout_equal(actual, desired, msg=msg)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return cast(np.ndarray, x.detach().cpu().numpy())


def assert_tensor_equal(
    actual: torch.Tensor,
    desired: torch.Tensor,
    assert_meta_equal: bool = True,
    verbose: bool = True,
    msg: Optional[str] = None,
) -> None:
    if assert_meta_equal:
        assert_tensor_meta_equal(actual, desired, msg=msg)
    elif actual.dtype.is_floating_point or desired.dtype.is_floating_point:  # type: ignore[attr-defined]
        msg = (
            "Due to the limitations of floating point arithmetic, comparing "
            "floating-point tensors for equality is not recommended. Use "
            "assert_tensor_allclose instead."
        )
        warnings.warn(msg, RuntimeWarning)

    np.testing.assert_equal(
        _to_numpy(actual), _to_numpy(desired), verbose=verbose, err_msg=msg
    )


def assert_tensor_allclose(
    actual: torch.Tensor,
    desired: torch.Tensor,
    assert_meta_equal: bool = True,
    rtol: float = 1e-7,
    atol: float = 0.0,
    equal_nan: bool = True,
    verbose: bool = True,
    msg: Optional[str] = None,
) -> None:
    if assert_meta_equal:
        assert_tensor_meta_equal(actual, desired, msg=msg)

    np.testing.assert_allclose(
        _to_numpy(actual),
        _to_numpy(desired),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        verbose=verbose,
        err_msg=msg,
    )


def approx(
    expected: Any,
    rel: Optional[float] = None,
    abs: Optional[float] = None,
    nan_ok: bool = False,
) -> ApproxBase:
    if isinstance(expected, torch.Tensor):
        expected = _to_numpy(expected)
    return pytest.approx(expected, rel=rel, abs=abs, nan_ok=nan_ok)
