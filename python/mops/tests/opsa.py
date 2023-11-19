import numpy as np
import pytest
from mops.reference_implementations import outer_product_scatter_add as ref_opsa

import mops
from mops import outer_product_scatter_add as opsa

np.random.seed(0xDEADBEEF)


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)
    P = np.sort(np.random.randint(10, size=(100)))
    n_O = 20
    return A, B, P, n_O


def test_opsa(valid_arguments):
    A, B, P, n_O = valid_arguments

    reference = ref_opsa(A, B, P, n_O)
    actual = opsa(A, B, P, n_O)
    assert np.allclose(reference, actual)


def test_opsa_no_neighbors(valid_arguments):
    A, B, P, n_O = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    P[P == 1] = 2

    reference = ref_opsa(A, B, P, n_O)
    actual = opsa(A, B, P, n_O)
    assert np.allclose(reference, actual)


def test_opsa_wrong_type(valid_arguments):
    A, B, P, n_O = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(TypeError, match="Wrong dtype for A in opsa: got int32"):
        opsa(A, B, P, n_O)


def test_opsa_wrong_number_of_dimensions(valid_arguments):
    A, B, P, n_O = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError, match="A must be a 2D array in opsa, got a 3D array"
    ):
        opsa(A, B, P, n_O)


def test_opsa_size_mismatch(valid_arguments):
    A, B, P, n_O = valid_arguments
    P = P[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and P along dimension 0 must match in opsa",
    ):
        opsa(A, B, P, n_O)


def test_opsa_out_of_bounds(valid_arguments):
    A, B, P, n_O = valid_arguments
    P[0] = A.shape[1]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array P in operation opsa contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        opsa(A, B, P, n_O)
