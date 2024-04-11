import mops.torch
import torch
from mops.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasaw,
)
from mops.torch.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasaw_torch,
)

torch.manual_seed(0xDEADBEEF)


def test_sasaw():
    A = torch.rand(100, 20)
    B = torch.rand(100, 200)
    W = torch.rand(25, 13, 200)
    C = torch.rand(50)
    indices_output_1 = torch.randint(25, size=(100,), dtype=torch.int32)
    indices_W_1 = torch.randint(25, size=(100,), dtype=torch.int32)
    output_size = 15
    indices_A = torch.randint(20, size=(50,), dtype=torch.int32)
    indices_W_2 = torch.randint(13, size=(50,), dtype=torch.int32)
    indices_output_2 = torch.randint(output_size, size=(50,), dtype=torch.int32)

    reference = torch.tensor(
        ref_sasaw(
            A.numpy(),
            B.numpy(),
            C.numpy(),
            W.numpy(),
            indices_A.numpy(),
            indices_W_1.numpy(),
            indices_W_2.numpy(),
            indices_output_1.numpy(),
            indices_output_2.numpy(),
            output_size,
        )
    )
    actual = mops.torch.sparse_accumulation_scatter_add_with_weights(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )
    assert torch.allclose(reference, actual)


def test_sasaw_grad():
    A = torch.rand(70, 10, dtype=torch.float64, requires_grad=True)
    B = torch.rand(70, 50, dtype=torch.float64, requires_grad=True)
    W = torch.rand(3, 5, 50, dtype=torch.float64, requires_grad=True)
    C = torch.rand(50, dtype=torch.float64)
    indices_output_1 = torch.randint(3, size=(70,), dtype=torch.int32)
    indices_W_1 = torch.randint(3, size=(70,), dtype=torch.int32)
    output_size_2 = 15
    indices_A = torch.randint(10, size=(50,), dtype=torch.int32)
    indices_W_2 = torch.randint(5, size=(50,), dtype=torch.int32)
    indices_output_2 = torch.randint(output_size_2, size=(50,), dtype=torch.int32)

    assert torch.autograd.gradcheck(
        mops.torch.sparse_accumulation_scatter_add_with_weights,
        (
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
            output_size_2,
        ),
        fast_mode=True,
    )


def test_sasaw_ref():
    A = torch.rand(70, 10)
    B = torch.rand(70, 50)
    W = torch.rand(3, 5, 50)
    C = torch.rand(50)
    indices_output_1 = torch.randint(3, size=(70,), dtype=torch.int32)
    indices_W_1 = torch.randint(3, size=(70,), dtype=torch.int32)
    output_size_2 = 15
    indices_A = torch.randint(10, size=(50,), dtype=torch.int32)
    indices_W_2 = torch.randint(5, size=(50,), dtype=torch.int32)
    indices_output_2 = torch.randint(output_size_2, size=(50,), dtype=torch.int32)

    reference = torch.tensor(
        ref_sasaw(
            A.numpy(),
            B.numpy(),
            C.numpy(),
            W.numpy(),
            indices_A.numpy(),
            indices_W_1.numpy(),
            indices_W_2.numpy(),
            indices_output_1.numpy(),
            indices_output_2.numpy(),
            output_size_2,
        )
    )
    actual = ref_sasaw_torch(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size_2,
    )
    assert torch.allclose(reference, actual)
