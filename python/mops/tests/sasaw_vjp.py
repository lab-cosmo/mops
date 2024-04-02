import numpy as np
import pytest
from mops.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights_vjp as ref_sasaw_vjp,
)

import mops
from mops import sparse_accumulation_scatter_add_with_weights_vjp as sasaw_vjp

np.random.seed(0xDEADBEEF)

try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    grad_output = np.random.rand(25, 15, 200)
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 200)
    W = np.random.rand(25, 13, 200)
    C = np.random.rand(50)
    indices_output_1 = np.random.randint(25, size=(100,))
    indices_W_1 = np.random.randint(25, size=(100,))
    output_size = 15
    indices_A = np.random.randint(20, size=(50,))
    indices_W_2 = np.random.randint(13, size=(50,))
    indices_output_2 = np.random.randint(output_size, size=(50,))

    return (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    )


def test_sasaw_vjp(valid_arguments):
    (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    ) = valid_arguments

    ref_grad_A, ref_grad_B, ref_grad_W = ref_sasaw_vjp(
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    )
    grad_A, grad_B, grad_W = sasaw_vjp(
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    )
    assert np.allclose(ref_grad_A, grad_A)
    assert np.allclose(ref_grad_B, grad_B)
    assert np.allclose(ref_grad_W, grad_W)


def test_sasaw_vjp_no_neighbors(valid_arguments):
    (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    ) = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output_1[indices_output_1 == 1] = 2

    ref_grad_A, ref_grad_B, ref_grad_W = ref_sasaw_vjp(
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    )
    grad_A, grad_B, grad_W = sasaw_vjp(
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    )
    assert np.allclose(ref_grad_A, grad_A)
    assert np.allclose(ref_grad_B, grad_B)
    assert np.allclose(ref_grad_W, grad_W)


def test_sasaw_vjp_wrong_type(valid_arguments):
    (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    ) = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError,
        match="Wrong dtype for A in "
        "sparse_accumulation_scatter_add_with_weights: got int32",
    ):
        sasaw_vjp(
            grad_output,
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
        )


def test_sasaw_vjp_wrong_number_of_dimensions(valid_arguments):
    (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    ) = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "sparse_accumulation_scatter_add_with_weights, got a 3D array instead",
    ):
        sasaw_vjp(
            grad_output,
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
        )


def test_sasaw_vjp_size_mismatch(valid_arguments):
    (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    ) = valid_arguments
    indices_output_1 = indices_output_1[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and indices_output_1 along dimension 0 must match in sasaw_vjp",
    ):
        sasaw_vjp(
            grad_output,
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
        )


def test_sasaw_vjp_out_of_bounds(valid_arguments):
    (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    ) = valid_arguments
    indices_output_1[0] = W.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_output_1 in operation "
        "sasaw_vjp contains elements up to 25; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 24",
    ):
        sasaw_vjp(
            grad_output,
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
        )


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_sasaw_vjp_cupy(valid_arguments):
    (
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    ) = valid_arguments
    grad_output = cp.array(grad_output)
    A = cp.array(A)
    B = cp.array(B)
    C = cp.array(C)
    W = cp.array(W)
    indices_A = cp.array(indices_A)
    indices_W_1 = cp.array(indices_W_1)
    indices_W_2 = cp.array(indices_W_2)
    indices_output_1 = cp.array(indices_output_1)
    indices_output_2 = cp.array(indices_output_2)

    ref_grad_A, ref_grad_B, ref_grad_W = ref_sasaw_vjp(  # noqa: F841
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
    )
    with pytest.raises(
        mops.status.MopsError, match="CUDA implementation does not exist yet"
    ):
        grad_A, grad_B, grad_W = sasaw_vjp(  # noqa: F841
            grad_output,
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
        )
