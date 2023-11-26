import mops.torch
import torch
from mops.reference_implementations import homogeneous_polynomial_evaluation as ref_hpe

torch.manual_seed(0xDEADBEEF)


def test_hpe():
    A = torch.rand(100, 20)
    C = torch.rand(200)
    indices_A = torch.randint(20, size=(200, 4), dtype=torch.int32)

    reference = torch.tensor(ref_hpe(A.numpy(), C.numpy(), indices_A.numpy()))
    actual = mops.torch.homogeneous_polynomial_evaluation(A, C, indices_A)
    assert torch.allclose(reference, actual)


def test_hpe_grad():
    A = torch.rand(100, 20, dtype=torch.float64, requires_grad=True)
    C = torch.rand(200, dtype=torch.float64)
    indices_A = torch.randint(20, size=(200, 4), dtype=torch.int32)

    assert torch.autograd.gradcheck(
        mops.torch.homogeneous_polynomial_evaluation,
        (A, C, indices_A),
        fast_mode=True,
    )
