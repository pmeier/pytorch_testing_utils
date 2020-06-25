import pytest
import torch

import pytorch_testing_utils as ptu


def test_assert_tensor_allclose():
    tensor1 = torch.sqrt(torch.tensor(2.0))
    tensor2 = torch.tensor(2.0) / torch.sqrt(torch.tensor(2.0))
    tensor3 = torch.tensor(577.0 / 408.0)

    ptu.assert_tensor_allclose(tensor1, tensor2)

    with pytest.raises(AssertionError):
        ptu.assert_tensor_allclose(tensor1, tensor3)

    ptu.assert_tensor_allclose(tensor1, tensor3, rtol=1e-5)


def test_assert_tensor_allclose_attributes():
    tensor1 = torch.tensor(1.0, dtype=torch.float32)
    tensor2 = torch.tensor(1, dtype=torch.int32)

    with pytest.raises(AssertionError):
        ptu.assert_tensor_allclose(tensor1, tensor2)

    ptu.assert_tensor_allclose(tensor1, tensor2, attributes=False)


def test_assert_tensor_equal_attributes():
    tensor1 = torch.tensor(1, dtype=torch.int32)
    tensor2 = torch.tensor(1, dtype=torch.int64)

    with pytest.raises(AssertionError):
        ptu.assert_tensor_equal(tensor1, tensor2)

    ptu.assert_tensor_equal(tensor1, tensor2, attributes=False)


def test_assert_tensor_equal_floating_point():
    tensor1 = torch.tensor(1, dtype=torch.int32)
    tensor2 = torch.tensor(1, dtype=torch.float32)

    try:
        with pytest.warns(RuntimeWarning):
            ptu.assert_tensor_equal(tensor1, tensor2, attributes=False)
    except AssertionError:
        pass


def test_approx():
    rel = 1e-6
    abs = 0.0
    approx_value = ptu.approx(torch.tensor(1.0), rel=rel, abs=abs)

    assert 1.0 + 0.9 * rel == approx_value
    assert approx_value == 1.0 + 0.9 * rel
