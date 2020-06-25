from collections import OrderedDict
from copy import copy

import pytest
import torch

import pytorch_testing_utils as ptu


def test_assert_tensor_seq_equal():
    seq = (torch.tensor(0), torch.tensor(1))
    ptu.assert_tensor_seq_equal(seq, copy(seq))


def test_assert_tensor_seq_equal_no_seq(subtests):
    value = torch.tensor(0)
    seq = (value,)
    nseq = {value}

    with subtests.test("desired no sequence"):
        with pytest.raises(AssertionError):
            ptu.assert_tensor_seq_equal(nseq, seq)

    with subtests.test("desired no sequence"):
        with pytest.raises(AssertionError):
            ptu.assert_tensor_seq_equal(seq, nseq)

    with subtests.test("both no sequence"):
        with pytest.raises(AssertionError):
            ptu.assert_tensor_seq_equal(nseq, nseq)


def test_assert_tensor_seq_equal_len_mismatch():
    seq1 = (torch.tensor(0),)
    seq2 = (torch.tensor(0), torch.tensor(1))

    with pytest.raises(AssertionError):
        ptu.assert_tensor_seq_equal(seq1, seq2)


def test_assert_tensor_seq_equal_item_mismatch():
    seq1 = (torch.tensor(0), torch.tensor(1))
    seq2 = (torch.tensor(0), torch.tensor(2))

    with pytest.raises(AssertionError):
        ptu.assert_tensor_seq_equal(seq1, seq2)


def test_assert_tensor_seq_allclose():
    atol = 1e-3
    seq1 = (torch.tensor(0.0), torch.tensor(1.0))
    seq2 = (torch.tensor(0.0 + atol / 2), torch.tensor(1.0 - atol / 2))
    ptu.assert_tensor_seq_allclose(seq1, seq2, rtol=0.0, atol=atol)


def test_assert_tensor_seq_allclose_item_mismatch():
    atol = 1e-3
    seq1 = (torch.tensor(0.0), torch.tensor(1.0))
    seq2 = (torch.tensor(0.0 + 2 * atol), torch.tensor(1.0 - 2 * atol))

    with pytest.raises(AssertionError):
        ptu.assert_tensor_seq_allclose(seq1, seq2, rtol=0.0, atol=atol)


def test_assert_tensor_dict_equal():
    dct = {"0": torch.tensor(0), "1:": torch.tensor(1)}
    ptu.assert_tensor_dict_equal(dct, copy(dct))


def test_assert_tensor_dict_equal_ordered_dict():
    dct = OrderedDict((("0", torch.tensor(0)), ("1", torch.tensor(1))))
    ptu.assert_equal(dct, copy(dct))


def test_assert_tensor_dict_equal_no_seq(subtests):
    value = torch.tensor(0)
    dct = (value,)
    ndct = {value}

    with subtests.test("desired no sequence"):
        with pytest.raises(AssertionError):
            ptu.assert_tensor_dict_equal(ndct, dct)

    with subtests.test("desired no sequence"):
        with pytest.raises(AssertionError):
            ptu.assert_tensor_dict_equal(dct, ndct)

    with subtests.test("both no sequence"):
        with pytest.raises(AssertionError):
            ptu.assert_tensor_dict_equal(ndct, ndct)


def test_assert_tensor_dict_equal_key_mismatch():
    dct1 = {"0": torch.tensor(0), "1:": torch.tensor(1)}
    dct2 = {"0": torch.tensor(0), "2:": torch.tensor(1)}

    with pytest.raises(AssertionError):
        ptu.assert_tensor_dict_equal(dct1, dct2)


@pytest.mark.skip("Ordering in OrderedDicts is currently not asserted.")
def test_assert_tensor_dict_equal_ordered_dict_key_mismatch():
    pairs = (("0", torch.tensor(0)), ("1", torch.tensor(1)))
    dct1 = OrderedDict(pairs)
    dct2 = OrderedDict(reversed(pairs))

    with pytest.raises(AssertionError):
        ptu.assert_tensor_dict_equal(dct1, dct2)


def test_assert_tensor_dict_equal_item_mismatch():
    dct1 = {"0": torch.tensor(0), "1:": torch.tensor(1)}
    dct2 = {"0": torch.tensor(0), "1:": torch.tensor(2)}

    with pytest.raises(AssertionError):
        ptu.assert_tensor_dict_equal(dct1, dct2)


def test_assert_tensor_dict_allclose():
    atol = 1e-3
    dct1 = {"0": torch.tensor(0.0), "1:": torch.tensor(1.0)}
    dct2 = {"0": torch.tensor(0.0 + atol / 2), "1:": torch.tensor(1.0 - atol / 2)}
    ptu.assert_tensor_dict_allclose(dct1, dct2, rtol=0.0, atol=atol)


def test_assert_tensor_dict_allclose_item_mismatch():
    atol = 1e-3
    dct1 = {"0": torch.tensor(0.0), "1:": torch.tensor(1.0)}
    dct2 = {"0": torch.tensor(0.0 + 2 * atol), "1:": torch.tensor(1.0 - 2 * atol)}

    with pytest.raises(AssertionError):
        ptu.assert_tensor_dict_allclose(dct1, dct2, rtol=0.0, atol=atol)
