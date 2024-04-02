import numpy as np
import pytest
from mops.reference_implementations import outer_product_scatter_add as ref_opsa

import mops
from mops import outer_product_scatter_add as opsa

try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 10)
    B = np.random.rand(100, 5)
    indices_output = np.random.randint(10, size=(100))
    output_size = 10
    return A, B, indices_output, output_size


def test_opsa(valid_arguments):
    A, B, indices_output, output_size = valid_arguments

    reference = ref_opsa(A, B, indices_output, output_size)
    actual = opsa(A, B, indices_output, output_size)
    assert np.allclose(reference, actual)


def test_opsa_no_neighbors(valid_arguments):
    A, B, indices_output, output_size = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output[indices_output == 1] = 2

    reference = ref_opsa(A, B, indices_output, output_size)
    actual = opsa(A, B, indices_output, output_size)
    assert np.allclose(reference, actual)


def test_opsa_wrong_type(valid_arguments):
    A, B, indices_output, output_size = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError, match="Wrong dtype for A in outer_product_scatter_add: got int32"
    ):
        opsa(A, B, indices_output, output_size)


def test_opsa_wrong_number_of_dimensions(valid_arguments):
    A, B, indices_output, output_size = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "outer_product_scatter_add, got a 3D array instead",
    ):
        opsa(A, B, indices_output, output_size)


def test_opsa_size_mismatch(valid_arguments):
    A, B, indices_output, output_size = valid_arguments
    indices_output = indices_output[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and indices_output along dimension 0 must match in opsa",
    ):
        opsa(A, B, indices_output, output_size)


def test_opsa_out_of_bounds(valid_arguments):
    A, B, indices_output, output_size = valid_arguments
    indices_output[0] = output_size

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_output in operation opsa "
        "contains elements up to 10; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 9",
    ):
        opsa(A, B, indices_output, output_size)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_opsa_cupy(valid_arguments):
    A, B, indices_output, output_size = valid_arguments
    indices_output = np.sort(indices_output)
    A = cp.array(A)
    B = cp.array(B)
    indices_output = cp.array(indices_output)

    reference = ref_opsa(A, B, indices_output, output_size)
    actual = opsa(A, B, indices_output, output_size)
    assert cp.allclose(reference, actual)
