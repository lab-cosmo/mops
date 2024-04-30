import mops.torch
import pytest
import torch
from mops.reference_implementations import sparse_accumulation_of_products as ref_sap
from mops.torch.reference_implementations import (
    sparse_accumulation_of_products as ref_sap_torch,
)

torch.manual_seed(0xDEADBEEF)

if torch.cuda.is_available():
    HAS_CUDA = True
else:
    HAS_CUDA = False


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sap(dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    A = torch.rand(99, 20, device=device, dtype=dtype)
    B = torch.rand(99, 6, device=device, dtype=dtype)
    C = torch.rand(30, device=device, dtype=dtype)
    indices_A = torch.randint(20, size=(30,), dtype=torch.int32, device=device)
    indices_B = torch.randint(6, size=(30,), dtype=torch.int32, device=device)
    output_size = 35
    indices_output = torch.randint(
        output_size, size=(30,), dtype=torch.int32, device=device
    )

    reference = torch.tensor(
        ref_sap(
            A.cpu().numpy(),
            B.cpu().numpy(),
            C.cpu().numpy(),
            indices_A.cpu().numpy(),
            indices_B.cpu().numpy(),
            indices_output.cpu().numpy(),
            output_size,
        ),
        dtype=dtype,
        device=device,
    )
    actual = mops.torch.sparse_accumulation_of_products(
        A, B, C, indices_A, indices_B, indices_output, output_size
    )
    assert torch.allclose(reference, actual)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sap_grads(device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    A = torch.rand(99, 20, device=device,
                   dtype=torch.float64, requires_grad=True)
    B = torch.rand(99, 6, device=device,
                   dtype=torch.float64, requires_grad=True)
    C = torch.rand(30, device=device, dtype=torch.float64)
    indices_A = torch.randint(20, size=(30,), dtype=torch.int32, device=device)
    indices_B = torch.randint(6, size=(30,), dtype=torch.int32, device=device)
    output_size = 35
    indices_output = torch.randint(
        output_size, size=(30,), dtype=torch.int32, device=device
    )

    assert torch.autograd.gradcheck(
        mops.torch.sparse_accumulation_of_products,
        (A, B, C, indices_A, indices_B, indices_output, output_size),
    )

    assert torch.autograd.gradgradcheck(
        mops.torch.sparse_accumulation_of_products,
        (A, B, C, indices_A, indices_B, indices_output, output_size),
    )


def test_sap_ref():
    A = torch.rand(99, 20)
    B = torch.rand(99, 6)
    C = torch.rand(30)
    indices_A = torch.randint(20, size=(30,), dtype=torch.int32)
    indices_B = torch.randint(6, size=(30,), dtype=torch.int32)
    output_size = 35
    indices_output = torch.randint(output_size, size=(30,), dtype=torch.int32)

    reference = torch.tensor(
        ref_sap(
            A.numpy(),
            B.numpy(),
            C.numpy(),
            indices_A.numpy(),
            indices_B.numpy(),
            indices_output.numpy(),
            output_size,
        )
    )
    actual = ref_sap_torch(A, B, C, indices_A, indices_B,
                           indices_output, output_size)
    assert torch.allclose(reference, actual)
