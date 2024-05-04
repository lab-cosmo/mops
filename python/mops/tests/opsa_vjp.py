import numpy as np
import pytest
from mops.reference_implementations import outer_product_scatter_add_vjp as ref_opsa_vjp

import mops
from mops import outer_product_scatter_add_vjp as opsa_vjp

try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    output_size = 10
    grad_output = np.random.rand(output_size, 20, 5)
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)
    indices_output = np.random.randint(10, size=(100))
    return grad_output, A, B, indices_output


def test_opsa_vjp(valid_arguments):
    grad_output, A, B, indices_output = valid_arguments

    ref_grad_A, ref_grad_B = ref_opsa_vjp(grad_output, A, B, indices_output)
    grad_A, grad_B = opsa_vjp(grad_output, A, B, indices_output)
    assert np.allclose(ref_grad_A, grad_A)
    assert np.allclose(ref_grad_B, grad_B)


def test_opsa_vjp_no_neighbors(valid_arguments):
    grad_output, A, B, indices_output = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output[indices_output == 1] = 2

    grad_A, grad_B = opsa_vjp(grad_output, A, B, indices_output)

    ref_grad_A, ref_grad_B = ref_opsa_vjp(grad_output, A, B, indices_output)

    assert np.allclose(grad_A, ref_grad_A)
    assert np.allclose(grad_B, ref_grad_B)


def test_opsa_vjp_wrong_type(valid_arguments):
    grad_output, A, B, indices_output = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError, match="Wrong dtype for A in outer_product_scatter_add: got int32"
    ):
        opsa_vjp(grad_output, A, B, indices_output)


def test_opsa_vjp_wrong_number_of_dimensions(valid_arguments):
    grad_output, A, B, indices_output = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "outer_product_scatter_add, got a 3D array instead",
    ):
        opsa_vjp(grad_output, A, B, indices_output)


def test_opsa_vjp_size_mismatch(valid_arguments):
    grad_output, A, B, indices_output = valid_arguments
    indices_output = indices_output[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and indices_output along dimension 0 must match in "
        "cpu_outer_product_scatter_add_vjp",
    ):
        opsa_vjp(grad_output, A, B, indices_output)


def test_opsa_vjp_out_of_bounds(valid_arguments):
    grad_output, A, B, indices_output = valid_arguments
    indices_output[0] = grad_output.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_output in operation "
        "cpu_outer_product_scatter_add_vjp "
        "contains elements up to 10; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 9",
    ):
        opsa_vjp(grad_output, A, B, indices_output)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_opsa_vjp_cupy(valid_arguments):
    grad_output, A, B, indices_output = valid_arguments
    indices_output = np.sort(indices_output)
    grad_output = cp.array(grad_output)
    A = cp.array(A)
    B = cp.array(B)
    indices_output = cp.array(indices_output)

    ref_grad_A, ref_grad_B = ref_opsa_vjp(grad_output, A, B, indices_output)
    grad_A, grad_B = opsa_vjp(grad_output, A, B, indices_output)
    assert cp.allclose(ref_grad_A, grad_A)
    assert cp.allclose(ref_grad_B, grad_B)
