import mops.torch
import torch
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsaw,
)

torch.manual_seed(0xDEADBEEF)


def test_opsaw():
    A = torch.rand(100, 10)
    B = torch.rand(100, 5)
    W = torch.rand(20, 5)
    indices_W = torch.randint(20, size=(100,), dtype=torch.int32)
    indices_output = torch.randint(20, size=(100,), dtype=torch.int32)

    reference = torch.tensor(
        ref_opsaw(
            A.numpy(), B.numpy(), W.numpy(), indices_W.numpy(), indices_output.numpy()
        )
    )
    actual = mops.torch.outer_product_scatter_add_with_weights(
        A, B, W, indices_W, indices_output
    )
    assert torch.allclose(reference, actual)


def test_opsaw_grad():
    A = torch.rand(100, 10, dtype=torch.float64, requires_grad=True)
    B = torch.rand(100, 5, dtype=torch.float64, requires_grad=True)
    n_O = 20
    W = torch.rand(n_O, 5, dtype=torch.float64, requires_grad=True)
    indices_W = torch.randint(20, size=(100,), dtype=torch.int32)
    indices_output = torch.randint(20, size=(100,), dtype=torch.int32)

    assert torch.autograd.gradcheck(
        mops.torch.outer_product_scatter_add_with_weights,
        (A, B, W, indices_W, indices_output),
        fast_mode=True,
    )
