import mops.torch
import pytest
import torch
from mops.reference_implementations import sparse_accumulation_of_products as ref_sap

torch.manual_seed(0xDEADBEEF)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sap(dtype, device):
    A = torch.rand(100, 20, device=device, dtype=dtype)
    B = torch.rand(100, 6, device=device, dtype=dtype)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sap_grad(dtype, device):
    A = torch.rand(100, 20, device=device, dtype=dtype, requires_grad=True)
    B = torch.rand(100, 6, device=device, dtype=dtype, requires_grad=True)
    C = torch.rand(30, device=device, dtype=dtype)
    indices_A = torch.randint(20, size=(30,), dtype=torch.int32, device=device)
    indices_B = torch.randint(6, size=(30,), dtype=torch.int32, device=device)
    output_size = 35
    indices_output = torch.randint(
        output_size, size=(30,), dtype=torch.int32, device=device
    )

    assert torch.autograd.gradcheck(
        mops.torch.sparse_accumulation_of_products,
        (A, B, C, indices_A, indices_B, indices_output, output_size),
        fast_mode=True,
    )
