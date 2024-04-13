import mops.torch
import pytest
import torch
from mops.reference_implementations import outer_product_scatter_add as ref_opsa
from mops.torch.reference_implementations import (
    outer_product_scatter_add as ref_opsa_torch,
)

torch.manual_seed(0xDEADBEEF)

if torch.cuda.is_available():
    HAS_CUDA = True
else:
    HAS_CUDA = False


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_opsa(dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    A = torch.rand(100, 20, dtype=dtype, device=device)
    B = torch.rand(100, 5, dtype=dtype, device=device)

    output_size = 10

    indices_output = torch.sort(
        torch.randint(output_size, size=(100,), dtype=torch.int32, device=device)
    ).values
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output[indices_output == 1] = 2

    reference = torch.tensor(
        ref_opsa(
            A.cpu().numpy(), B.cpu().numpy(), indices_output.cpu().numpy(), output_size
        ),
        dtype=dtype,
        device=device,
    )

    actual = mops.torch.outer_product_scatter_add(A, B, indices_output, output_size)

    assert torch.allclose(reference, actual)


@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_opsa_grads(dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    A = torch.rand(100, 20, dtype=dtype, requires_grad=True, device=device)
    B = torch.rand(100, 5, dtype=dtype, requires_grad=True, device=device)

    output_size = 10
    indices = torch.sort(
        torch.randint(output_size, size=(100,), dtype=torch.int32, device=device)
    ).values

    assert torch.autograd.gradcheck(
        mops.torch.outer_product_scatter_add,
        (A, B, indices, output_size),
    )

    if device != "cuda":  # not yet implemented
        assert torch.autograd.gradgradcheck(
            mops.torch.outer_product_scatter_add,
            (A, B, indices, output_size),
        )


def test_opsa_ref():
    A = torch.rand(100, 20)
    B = torch.rand(100, 5)

    output_size = 10

    indices_output = torch.sort(
        torch.randint(output_size, size=(100,), dtype=torch.int32)
    ).values
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output[indices_output == 1] = 2

    reference = torch.tensor(
        ref_opsa(
            A.cpu().numpy(), B.cpu().numpy(), indices_output.cpu().numpy(), output_size
        )
    )

    actual = ref_opsa_torch(A, B, indices_output, output_size)

    assert torch.allclose(reference, actual)
