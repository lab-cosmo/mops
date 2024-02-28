import mops.torch
import torch
import pytest
from time import time
from mops import reference_implementations as ref

torch.manual_seed(0xDEADBEEF)


def test_opsa():
    A = torch.rand(100, 20)
    B = torch.rand(100, 5)

    output_size = 10

    indices = torch.sort(
        torch.randint(output_size, size=(100,), dtype=torch.int32)
    ).values
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices[indices == 1] = 2

    reference = torch.tensor(
        ref.outer_product_scatter_add(
            A.numpy(), B.numpy(), indices.numpy(), output_size
        )
    )
    actual = mops.torch.outer_product_scatter_add(A, B, indices, output_size)
    assert torch.allclose(reference, actual)


def test_opsa_grad():
    A = torch.rand(100, 20, dtype=torch.float64, requires_grad=True)
    B = torch.rand(100, 5, dtype=torch.float64, requires_grad=True)

    output_size = 10
    indices = torch.sort(
        torch.randint(output_size, size=(100,), dtype=torch.int32)
    ).values

    assert torch.autograd.gradcheck(
        mops.torch.outer_product_scatter_add,
        (A, B, indices, output_size),
        fast_mode=True,
    )


def test_opsa_cuda():
    A = torch.rand(100, 20)
    B = torch.rand(100, 5)

    output_size = 10

    indices = torch.sort(
        torch.randint(output_size, size=(100,), dtype=torch.int32)
    ).values
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices[indices == 1] = 2

    reference = torch.tensor(
        ref.outer_product_scatter_add(
            A.numpy(), B.numpy(), indices.numpy(), output_size
        )
    )

    actual = mops.torch.outer_product_scatter_add(
        A.cuda(), B.cuda(), indices.cuda(), output_size)

    assert torch.allclose(reference, actual.cpu())
    
def test_opsa_grad_cuda():
    A = torch.rand(100, 20, dtype=torch.float64, requires_grad=True, device='cuda')
    B = torch.rand(100, 5, dtype=torch.float64, requires_grad=True, device='cuda')

    output_size = 10
    indices = torch.sort(
        torch.randint(output_size, size=(100,), dtype=torch.int32)
    ).values.cuda()

    assert torch.autograd.gradcheck(
        mops.torch.outer_product_scatter_add,
        (A, B, indices, output_size),
        fast_mode=True,
    )