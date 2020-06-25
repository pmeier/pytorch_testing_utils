from typing import Any, Optional

import torch

__all__ = [
    "assert_tensor_dtype_equal",
    "assert_tensor_layout_equal",
    "assert_tensor_device_equal",
    # "assert_memory_format_equal",
    "assert_tensor_attributes_equal",
]


def _assert_equal(
    actual: Any, desired: Any, msg: Optional[str] = None, prefix: Optional[str] = None
) -> None:
    if msg is None:
        msg = f"{actual} != {desired}"
        if prefix is not None:
            msg = prefix + msg

    assert actual == desired, msg


def _assert_attr_equal(
    attr: str, actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None
) -> None:
    _assert_equal(
        getattr(actual, attr),
        getattr(desired, attr),
        msg=msg,
        prefix=f"{attr} mismatch: ",
    )


def assert_tensor_dtype_equal(
    actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
) -> None:
    _assert_attr_equal("dtype", actual, desired, msg=msg)


def assert_tensor_layout_equal(
    actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
) -> None:
    _assert_attr_equal("layout", actual, desired, msg=msg)


def assert_tensor_device_equal(
    actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
) -> None:
    _assert_attr_equal("device", actual, desired, msg=msg)


# FIXME: Tensor.memory_format is not a valid attribute.
# def assert_tensor_memory_format_equal(
#     actual: torch.Tensor, desired: torch.Tensor, msg: Optional[str] = None,
# ) -> None:
#     _assert_attr_equal("memory_format", actual, desired, msg=msg)


def assert_tensor_attributes_equal(
    actual: torch.Tensor,
    desired: torch.Tensor,
    dtype: bool = True,
    device: bool = True,
    layout: bool = True,
    # memory_format: bool = True,
    msg: Optional[str] = None,
) -> None:
    if dtype:
        assert_tensor_dtype_equal(actual, desired, msg=msg)
    if device:
        assert_tensor_device_equal(actual, desired, msg=msg)
    if layout:
        assert_tensor_layout_equal(actual, desired, msg=msg)
    # FIXME: Enable this when assert_tensor_memory_format_equal is implemented.
    # if memory_format:
    #     assert_tensor_memory_format_equal(actual, desired, msg=msg)
