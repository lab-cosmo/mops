import mops.torch
import pytest
import torch
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsaw,
)
from mops.torch.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsaw_torch,
)

torch.manual_seed(0xDEADBEEF)

if torch.cuda.is_available():
    HAS_CUDA = True
else:
    HAS_CUDA = False


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


@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_opsaw_grads(dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    A = torch.rand(100, 10, dtype=dtype, device=device, requires_grad=True)
    B = torch.rand(100, 5, dtype=dtype, device=device, requires_grad=True)
    n_O = 20
    W = torch.rand(n_O, 5, dtype=dtype, device=device, requires_grad=True)
    indices_W = torch.randint(20, size=(100,), device=device, dtype=torch.int32)
    indices_output = torch.randint(20, size=(100,), device=device, dtype=torch.int32)

    if device != "cuda":  # not yet implemented
        assert torch.autograd.gradcheck(
            mops.torch.outer_product_scatter_add_with_weights,
            (A, B, W, indices_W, indices_output),
        )

    # not yet implemented
    # assert torch.autograd.gradgradcheck(
    #     mops.torch.outer_product_scatter_add_with_weights,
    #     (A, B, W, indices_W, indices_output),
    # )


def test_opsaw_ref():
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
    actual = ref_opsaw_torch(A, B, W, indices_W, indices_output)
    assert torch.allclose(reference, actual)
