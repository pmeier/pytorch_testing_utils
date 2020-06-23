import itertools

import pytest
import torch

import pytorch_testing_utils as ptu

from .marks import skip_if_cuda_not_available


def test_assert_tensor_size_equal(subtests):
    sizes = [torch.Size(range(1, dims + 1)) for dims in range(1, 5)]

    for size in sizes:
        tensor1 = torch.empty(size)
        tensor2 = torch.empty(size)
        with subtests.test(size1=size, size2=size):
            ptu.assert_tensor_size_equal(tensor1, tensor2)

    for size1, size2 in itertools.permutations(sizes, 2):
        tensor1 = torch.empty(size1)
        tensor2 = torch.empty(size2)
        with subtests.test(size1=size1, size2=size2):
            with pytest.raises(AssertionError):
                ptu.assert_tensor_size_equal(tensor1, tensor2)


def test_assert_tensor_dtype_equal(subtests):
    dtypes = (
        torch.float32,
        torch.float64,
        torch.float16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )

    for dtype in dtypes:
        tensor1 = torch.empty(1, dtype=dtype)
        tensor2 = torch.empty(1, dtype=dtype)
        with subtests.test(dtype1=dtype, dtype2=dtype):
            ptu.assert_tensor_dtype_equal(tensor1, tensor2)

    for dtype1, dtype2 in itertools.permutations(dtypes, 2):
        tensor1 = torch.empty(1, dtype=dtype1)
        tensor2 = torch.empty(1, dtype=dtype2)
        with subtests.test(dtype1=dtype1, dtype2=dtype2):
            with pytest.raises(AssertionError):
                ptu.assert_tensor_dtype_equal(tensor1, tensor2)


def test_assert_tensor_layout_equal(subtests):
    layouts = (torch.strided, torch.sparse_coo)

    for layout in layouts:
        tensor1 = torch.empty(1, layout=layout)
        tensor2 = torch.empty(1, layout=layout)
        with subtests.test(layout1=layout, layout2=layout):
            ptu.assert_tensor_layout_equal(tensor1, tensor2)

    for layout1, layout2 in itertools.permutations(layouts, 2):
        tensor1 = torch.empty(1, layout=layout1)
        tensor2 = torch.empty(1, layout=layout2)
        with subtests.test(layout1=layout1, layout2=layout2):
            with pytest.raises(AssertionError):
                ptu.assert_tensor_layout_equal(tensor1, tensor2)


@skip_if_cuda_not_available
def test_assert_tensor_device_equal(subtests):
    devices = (
        torch.device("cpu"),
        *[
            torch.device("cuda", ordinal)
            for ordinal in range(torch.cuda.device_count())
        ],
    )

    for device in devices:
        tensor1 = torch.empty(1, device=device)
        tensor2 = torch.empty(1, device=device)
        with subtests.test(device1=device, device2=device):
            ptu.assert_tensor_device_equal(tensor1, tensor2)

    for device1, device2 in itertools.permutations(devices, 2):
        tensor1 = torch.empty(1, device=device1)
        tensor2 = torch.empty(1, device=device2)
        with subtests.test(device1=device1, device2=device2):
            with pytest.raises(AssertionError):
                ptu.assert_tensor_device_equal(tensor1, tensor2)


# FIXME: Enable this when ptu.assert_tensor_memory_format_equal is implemented.
# def test_assert_tensor_memory_format_equal(subtests):
#     memory_formats = (
#         torch.contiguous_format,
#         torch.channels_last,
#         torch.preserve_format,
#     )
#
#     for memory_format in memory_formats:
#         tensor1 = torch.empty(1, memory_format=memory_format)
#         tensor2 = torch.empty(1, memory_format=memory_format)
#         with subtests.test(memory_format1=memory_format, memory_format2=memory_format):
#             ptu.assert_tensor_memory_format_equal(tensor1, tensor2)
#
#     for memory_format1, memory_format2 in itertools.permutations(memory_formats, 2):
#         tensor1 = torch.empty(1, memory_format=memory_format1)
#         tensor2 = torch.empty(1, memory_format=memory_format2)
#         with subtests.test(
#             memory_format1=memory_format1, memory_format2=memory_format2
#         ):
#             with pytest.raises(AssertionError):
#                 ptu.assert_tensor_memory_format_equal(tensor1, tensor2)


def test_assert_tensor_allclose():
    tensor1 = torch.sqrt(torch.tensor(2.0))
    tensor2 = torch.tensor(2.0) / torch.sqrt(torch.tensor(2.0))
    tensor3 = torch.tensor(577.0 / 408.0)

    ptu.assert_tensor_allclose(tensor1, tensor2)

    with pytest.raises(AssertionError):
        ptu.assert_tensor_allclose(tensor1, tensor3)

    ptu.assert_tensor_allclose(tensor1, tensor3, rtol=1e-5)


def test_assert_tensor_allclose_meta():
    tensor1 = torch.tensor(1.0, dtype=torch.float32)
    tensor2 = torch.tensor(1, dtype=torch.int32)

    with pytest.raises(AssertionError):
        ptu.assert_tensor_allclose(tensor1, tensor2)

    ptu.assert_tensor_allclose(tensor1, tensor2, assert_meta_equal=False)


def test_assert_tensor_equal_meta():
    tensor1 = torch.tensor(1, dtype=torch.int32)
    tensor2 = torch.tensor(1, dtype=torch.int64)

    with pytest.raises(AssertionError):
        ptu.assert_tensor_equal(tensor1, tensor2)

    ptu.assert_tensor_equal(tensor1, tensor2, assert_meta_equal=False)


def test_assert_tensor_equal_floating_point():
    tensor1 = torch.tensor(1, dtype=torch.int32)
    tensor2 = torch.tensor(1, dtype=torch.float32)

    try:
        with pytest.warns(RuntimeWarning):
            ptu.assert_tensor_equal(tensor1, tensor2, assert_meta_equal=False)
    except AssertionError:
        pass
