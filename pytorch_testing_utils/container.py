from collections.abc import Sequence
from typing import Any, Callable, Dict
from typing import Sequence as SequenceType

import torch

from .values import _tensor_allclose_asserter, _tensor_equal_asserter

__all__ = [
    "assert_tensor_seq_equal",
    "assert_tensor_seq_allclose",
    "assert_tensor_dict_equal",
    "assert_tensor_dict_allclose",
]


def _assert_tensor_seq(
    tensor_asserter: Callable[[torch.Tensor, torch.Tensor], None],
    actual: SequenceType[torch.Tensor],
    desired: SequenceType[torch.Tensor],
) -> None:
    assert isinstance(actual, Sequence)
    assert isinstance(desired, Sequence)

    assert len(actual) == len(desired)
    for actual_item, desired_item in zip(actual, desired):
        tensor_asserter(actual_item, desired_item)


def assert_tensor_seq_equal(
    actual: SequenceType[torch.Tensor],
    desired: SequenceType[torch.Tensor],
    **tensor_equal_kwargs: Any,
) -> None:
    _assert_tensor_seq(_tensor_equal_asserter(**tensor_equal_kwargs), actual, desired)


def assert_tensor_seq_allclose(
    actual: SequenceType[torch.Tensor],
    desired: SequenceType[torch.Tensor],
    **tensor_allclose_kwargs: Any,
) -> None:
    _assert_tensor_seq(
        _tensor_allclose_asserter(**tensor_allclose_kwargs), actual, desired
    )


def _assert_tensor_dict(
    tensor_asserter: Callable[[torch.Tensor, torch.Tensor], None],
    actual: Dict[Any, torch.Tensor],
    desired: Dict[Any, torch.Tensor],
) -> None:
    assert isinstance(actual, dict)
    assert isinstance(desired, dict)

    assert actual.keys() == desired.keys()
    for key, actual_item in actual.items():
        tensor_asserter(actual_item, desired[key])


def assert_tensor_dict_equal(
    actual: Dict[Any, torch.Tensor],
    desired: Dict[Any, torch.Tensor],
    **tensor_equal_kwargs: Any,
) -> None:
    _assert_tensor_dict(_tensor_equal_asserter(**tensor_equal_kwargs), actual, desired)


def assert_tensor_dict_allclose(
    actual: Dict[Any, torch.Tensor],
    desired: Dict[Any, torch.Tensor],
    **tensor_allclose_kwargs: Any,
) -> None:
    _assert_tensor_dict(
        _tensor_allclose_asserter(**tensor_allclose_kwargs), actual, desired
    )
