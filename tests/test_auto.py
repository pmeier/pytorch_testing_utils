from collections import OrderedDict
from copy import copy

import numpy as np
import pytest
import torch

import pytorch_testing_utils as ptu


def test_assert_equal_tensor():
    tensor = torch.tensor(0)
    ptu.assert_equal(tensor, tensor.clone())


def test_assert_equal_tensor_seq():
    seq = (torch.tensor(0), torch.tensor(1))
    ptu.assert_equal(seq, copy(seq))


def test_assert_equal_tensor_dict():
    dct = {"0": torch.tensor(0), "1": torch.tensor(1)}
    ptu.assert_equal(dct, copy(dct))


def test_assert_equal_type_mismatch_tensor():
    tensor1 = torch.tensor(0)
    tensor2 = np.array(0)

    with pytest.raises(AssertionError):
        ptu.assert_equal(tensor1, tensor2)


def test_assert_equal_type_mismatch_seq():
    value = torch.tensor(0)
    seq1 = (value,)
    seq2 = [value]

    with pytest.raises(AssertionError):
        ptu.assert_equal(seq1, seq2)


def test_assert_equal_type_mismatch_dict():
    value = torch.tensor(0)
    dct1 = {"0": value}
    dct2 = OrderedDict((("0", value),))

    with pytest.raises(AssertionError):
        ptu.assert_equal(dct1, dct2)


def test_assert_equal_unknown_container():
    value = torch.tensor(0)
    st = {value}

    with pytest.raises(RuntimeError):
        ptu.assert_equal(st, copy(st))


def test_assert_allclose_tensor():
    atol = 1e-3
    tensor1 = torch.tensor(0.0)
    tensor2 = torch.tensor(0.0 + atol / 2)
    ptu.assert_allclose(tensor1, tensor2, rtol=0.0, atol=1e-3)


def test_assert_allclose_tensor_seq():
    atol = 1e-3
    seq1 = (torch.tensor(0.0), torch.tensor(1.0))
    seq2 = (torch.tensor(0.0 + atol / 2), torch.tensor(1.0 - atol / 2))
    ptu.assert_allclose(seq1, seq2, rtol=0.0, atol=1e-3)


def test_assert_allclose_tensor_dict():
    atol = 1e-3
    dct1 = {"0": torch.tensor(0.0), "1:": torch.tensor(1.0)}
    dct2 = {"0": torch.tensor(0.0 + atol / 2), "1:": torch.tensor(1.0 - atol / 2)}
    ptu.assert_allclose(dct1, dct2, rtol=0.0, atol=atol)
