import mops.status
import numpy as np
import pytest
from mops.reference_implementations import (
    sparse_accumulation_of_products_vjp_vjp as ref_sap_vjp_vjp,
)

import mops
from mops import sparse_accumulation_of_products_vjp_vjp as sap_vjp_vjp

np.random.seed(0xDEADBEEF)


try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    grad_grad_A = np.random.rand(99, 20)
    grad_grad_B = np.random.rand(99, 6)
    grad_output = np.random.rand(99, 35)
    A = np.random.rand(99, 20)
    B = np.random.rand(99, 6)
    C = np.random.rand(30)
    indices_A = np.random.randint(20, size=(30,))
    indices_B = np.random.randint(6, size=(30,))
    output_size = 35
    indices_output = np.random.randint(output_size, size=(30,))
    return (
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    )


def test_sap_vjp_vjp(valid_arguments):
    (
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    ) = valid_arguments

    ref = ref_sap_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    )
    actual = sap_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    )
    for i in range(len(ref)):
        assert np.allclose(ref[i], actual[i])


def test_sap_vjp_vjp_wrong_type(valid_arguments):
    (
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    ) = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError,
        match="Wrong dtype for A in sparse_accumulation_of_products: got int32",
    ):
        sap_vjp_vjp(
            grad_grad_A,
            grad_grad_B,
            grad_output,
            A,
            B,
            C,
            indices_A,
            indices_B,
            indices_output,
        )


def test_sap_vjp_vjp_wrong_number_of_dimensions(valid_arguments):
    (
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    ) = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "sparse_accumulation_of_products, got a 3D array instead",
    ):
        sap_vjp_vjp(
            grad_grad_A,
            grad_grad_B,
            grad_output,
            A,
            B,
            C,
            indices_A,
            indices_B,
            indices_output,
        )


def test_sap_vjp_vjp_size_mismatch(valid_arguments):
    (
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    ) = valid_arguments
    C = C[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of C along "
        "dimension 0 and indices_A along dimension 0 must match in "
        "cpu_sparse_accumulation_of_products_vjp_vjp",
    ):
        sap_vjp_vjp(
            grad_grad_A,
            grad_grad_B,
            grad_output,
            A,
            B,
            C,
            indices_A,
            indices_B,
            indices_output,
        )


def test_sap_vjp_vjp_out_of_bounds(valid_arguments):
    (
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    ) = valid_arguments
    indices_A[0] = A.shape[1]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_A in operation "
        "cpu_sparse_accumulation_of_products_vjp_vjp contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        sap_vjp_vjp(
            grad_grad_A,
            grad_grad_B,
            grad_output,
            A,
            B,
            C,
            indices_A,
            indices_B,
            indices_output,
        )


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_sap_vjp_vjp_cupy(valid_arguments):
    (
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    ) = valid_arguments
    grad_grad_A = cp.array(grad_grad_A)
    grad_grad_B = cp.array(grad_grad_B)
    grad_output = cp.array(grad_output)
    A = cp.array(A)
    B = cp.array(B)
    C = cp.array(C)
    indices_A = cp.array(indices_A)
    indices_B = cp.array(indices_B)
    indices_output = cp.array(indices_output)

    reference = ref_sap_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    )

    actual = sap_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
    )

    for i in range(len(reference)):
        assert cp.allclose(reference[i], actual[i])
