import numpy as np
import pytest
from mops.reference_implementations import homogeneous_polynomial_evaluation as ref_hpe

import mops
from mops import homogeneous_polynomial_evaluation as hpe

np.random.seed(0xDEADBEEF)


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 20)
    C = np.random.rand(200)
    P = np.random.randint(20, size=(200, 4))
    return A, C, P


def test_hpe(valid_arguments):
    A, C, P = valid_arguments

    reference = ref_hpe(A, C, P)
    actual = hpe(A, C, P)
    assert np.allclose(reference, actual)


def test_hpe_wrong_type(valid_arguments):
    A, C, P = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(TypeError, match="Wrong dtype for A in hpe: got int32"):
        hpe(A, C, P)


def test_hpe_wrong_number_of_dimensions(valid_arguments):
    A, C, P = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(ValueError, match="A must be a 2D array in hpe, got a 3D array"):
        hpe(A, C, P)


def test_hpe_size_mismatch(valid_arguments):
    A, C, P = valid_arguments
    C = C[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of C along "
        "dimension 0 and P along dimension 0 must match in hpe",
    ):
        hpe(A, C, P)


def test_hpe_out_of_bounds(valid_arguments):
    A, C, P = valid_arguments
    P[0] = A.shape[1]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array P in operation hpe contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        hpe(A, C, P)
