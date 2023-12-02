import mops.torch
import torch
from mops.reference_implementations import sparse_accumulation_of_products as ref_sap

torch.manual_seed(0xDEADBEEF)


def test_sap():
    A = torch.rand(100, 20)
    B = torch.rand(100, 6)
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
    actual = mops.torch.sparse_accumulation_of_products(
        A, B, C, indices_A, indices_B, indices_output, output_size
    )
    assert torch.allclose(reference, actual)


def test_sap_grad():
    A = torch.rand(100, 20, dtype=torch.float64, requires_grad=True)
    B = torch.rand(100, 6, dtype=torch.float64, requires_grad=True)
    C = torch.rand(30, dtype=torch.float64)
    indices_A = torch.randint(20, size=(30,), dtype=torch.int32)
    indices_B = torch.randint(6, size=(30,), dtype=torch.int32)
    output_size = 35
    indices_output = torch.randint(output_size, size=(30,), dtype=torch.int32)

    assert torch.autograd.gradcheck(
        mops.torch.sparse_accumulation_of_products,
        (A, B, C, indices_A, indices_B, indices_output, output_size),
        fast_mode=True,
    )
