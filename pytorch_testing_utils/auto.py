from typing import Any, Callable

import torch

from .container import _assert_tensor_dict, _assert_tensor_seq
from .values import _tensor_allclose_asserter, _tensor_equal_asserter

__all__ = ["assert_equal", "assert_allclose"]


def assert_equal(actual: Any, desired: Any, **tensor_equal_kwargs: Any,) -> None:
    _assert_by_type(
        _tensor_equal_asserter(**tensor_equal_kwargs,), actual, desired,
    )


def assert_allclose(actual: Any, desired: Any, **tensor_allclose_kwargs: Any,) -> None:
    _assert_by_type(
        _tensor_allclose_asserter(**tensor_allclose_kwargs), actual, desired,
    )


from collections.abc import Sequence


def _assert_by_type(
    tensor_asserter: Callable[[torch.Tensor, torch.Tensor], None],
    actual: Any,
    desired: Any,
) -> None:
    assert type(actual) == type(desired)

    if isinstance(actual, torch.Tensor):
        tensor_asserter(actual, desired)
        return

    asserter: Callable[[Callable[[torch.Tensor, torch.Tensor], None], Any, Any], None]
    if isinstance(actual, Sequence):
        asserter = _assert_tensor_seq
    elif isinstance(desired, dict):
        asserter = _assert_tensor_dict
    else:
        raise RuntimeError

    asserter(tensor_asserter, actual, desired)
