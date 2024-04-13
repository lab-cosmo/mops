import mops.torch
import pytest
import torch
from mops.reference_implementations import homogeneous_polynomial_evaluation as ref_hpe
from mops.torch.reference_implementations import (
    homogeneous_polynomial_evaluation as ref_hpe_torch,
)

torch.manual_seed(0xDEADBEEF)

if torch.cuda.is_available():
    HAS_CUDA = True
else:
    HAS_CUDA = False


@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hpe(dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    A = torch.rand(100, 20, dtype=dtype, device=device)
    C = torch.rand(200, dtype=dtype, device=device)
    indices_A = torch.randint(20, size=(200, 4), dtype=torch.int32, device=device)

    reference = torch.tensor(
        ref_hpe(A.cpu().numpy(), C.cpu().numpy(), indices_A.cpu().numpy()),
        dtype=torch.float64,
        device=device,
    )
    actual = mops.torch.homogeneous_polynomial_evaluation(A, C, indices_A)
    assert torch.allclose(reference, actual)


@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hpe_grads(dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    A = torch.rand(100, 20, dtype=dtype, device=device, requires_grad=True)
    C = torch.rand(200, dtype=dtype, device=device)
    indices_A = torch.randint(20, size=(200, 4), dtype=torch.int32, device=device)

    assert torch.autograd.gradcheck(
        mops.torch.homogeneous_polynomial_evaluation,
        (A, C, indices_A),
        nondet_tol=1e-5,
    )

    if device != "cuda":  # not yet implemented
        assert torch.autograd.gradgradcheck(
            mops.torch.homogeneous_polynomial_evaluation,
            (A, C, indices_A),
        )


def test_hpe_ref():
    A = torch.rand(100, 20)
    C = torch.rand(200)
    indices_A = torch.randint(20, size=(200, 4), dtype=torch.int32)

    reference = torch.tensor(
        ref_hpe(A.cpu().numpy(), C.cpu().numpy(), indices_A.cpu().numpy())
    )
    actual = ref_hpe_torch(A, C, indices_A)
    assert torch.allclose(reference, actual)
