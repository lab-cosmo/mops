import numpy as np
import pytest
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights_vjp as ref_opsaw_vjp,
)

import mops
from mops import outer_product_scatter_add_with_weights_vjp as opsaw_vjp

np.random.seed(0xDEADBEEF)

try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    grad_output = np.random.rand(20, 10, 5)
    A = np.random.rand(100, 10)
    B = np.random.rand(100, 5)
    W = np.random.rand(20, 5)
    indices_W = np.random.randint(20, size=(100,))
    indices_output = np.random.randint(20, size=(100,))

    return grad_output, A, B, W, indices_W, indices_output


def test_opsaw_vjp(valid_arguments):
    grad_output, A, B, W, indices_W, indices_output = valid_arguments

    ref_grad_A, ref_grad_B, ref_grad_W = ref_opsaw_vjp(
        grad_output, A, B, W, indices_W, indices_output
    )
    grad_A, grad_B, grad_W = opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)
    assert np.allclose(ref_grad_A, grad_A)
    assert np.allclose(ref_grad_B, grad_B)
    assert np.allclose(ref_grad_W, grad_W)


def test_opsaw_vjp_no_neighbors(valid_arguments):
    grad_output, A, B, W, indices_W, indices_output = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output[indices_output == 1] = 2

    ref_grad_A, ref_grad_B, ref_grad_W = ref_opsaw_vjp(
        grad_output, A, B, W, indices_W, indices_output
    )
    grad_A, grad_B, grad_W = opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)
    assert np.allclose(ref_grad_A, grad_A)
    assert np.allclose(ref_grad_B, grad_B)
    assert np.allclose(ref_grad_W, grad_W)


def test_opsaw_vjp_wrong_type(valid_arguments):
    grad_output, A, B, W, indices_W, indices_output = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError,
        match="Wrong dtype for A in "
        "outer_product_scatter_add_with_weights: got int32",
    ):
        opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)


def test_opsaw_vjp_wrong_number_of_dimensions(valid_arguments):
    grad_output, A, B, W, indices_W, indices_output = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "outer_product_scatter_add_with_weights, got a 3D array instead",
    ):
        opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)


def test_opsaw_vjp_size_mismatch(valid_arguments):
    grad_output, A, B, W, indices_W, indices_output = valid_arguments
    indices_output = indices_output[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and indices_output along dimension 0 must match in opsaw_vjp",
    ):
        opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)


def test_opsaw_vjp_out_of_bounds(valid_arguments):
    grad_output, A, B, W, indices_W, indices_output = valid_arguments
    indices_output[0] = W.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_output in operation opsaw_vjp "
        "contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_opsaw_vjp_cupy(valid_arguments):
    grad_output, A, B, W, indices_W, indices_output = valid_arguments
    grad_output = cp.array(grad_output)
    A = cp.array(A)
    B = cp.array(B)
    W = cp.array(W)
    indices_W = cp.array(indices_W)
    indices_output = cp.array(indices_output)

    ref_grad_A, ref_grad_B, ref_grad_W = ref_opsaw_vjp(  # noqa: F841
        grad_output, A, B, W, indices_W, indices_output
    )
    with pytest.raises(
        mops.status.MopsError, match="CUDA implementation does not exist yet"
    ):
        grad_A, grad_B, grad_W = opsaw_vjp(
            grad_output, A, B, W, indices_W, indices_output
        )  # noqa: F841
