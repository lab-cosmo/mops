import numpy as np
import pytest
from mops.reference_implementations import homogeneous_polynomial_evaluation as ref_hpe

import mops
from mops import homogeneous_polynomial_evaluation as hpe

np.random.seed(0xDEADBEEF)


try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    A = np.random.rand(99, 20)
    C = np.random.rand(200)
    indices_A = np.random.randint(20, size=(200, 4))
    return A, C, indices_A


def test_hpe(valid_arguments):
    A, C, indices_A = valid_arguments

    reference = ref_hpe(A, C, indices_A)
    actual = hpe(A, C, indices_A)
    assert np.allclose(reference, actual)


def test_hpe_wrong_type(valid_arguments):
    A, C, indices_A = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError,
        match="Wrong dtype for A in " "homogeneous_polynomial_evaluation: got int32",
    ):
        hpe(A, C, indices_A)


def test_hpe_wrong_number_of_dimensions(valid_arguments):
    A, C, indices_A = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "homogeneous_polynomial_evaluation, got a 3D array instead",
    ):
        hpe(A, C, indices_A)


def test_hpe_size_mismatch(valid_arguments):
    A, C, indices_A = valid_arguments
    C = C[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of C along "
        "dimension 0 and indices_A along dimension 0 must match in "
        "cpu_homogeneous_polynomial_evaluation",
    ):
        hpe(A, C, indices_A)


def test_hpe_out_of_bounds(valid_arguments):
    A, C, indices_A = valid_arguments
    indices_A[0] = A.shape[1]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_A in operation "
        "cpu_homogeneous_polynomial_evaluation "
        "contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        hpe(A, C, indices_A)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_hpe_cupy(valid_arguments):
    A, C, indices_A = valid_arguments

    A = cp.array(A)
    C = cp.array(C)
    indices_A = cp.array(indices_A)

    reference = ref_hpe(A, C, indices_A)
    actual = hpe(A, C, indices_A)

    assert cp.allclose(reference, actual)
