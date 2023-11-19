import numpy as np
import pytest
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsax,
)

import mops
from mops import outer_product_scatter_add_with_weights as opsax

np.random.seed(0xDEADBEEF)


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 10)
    R = np.random.rand(100, 5)
    X = np.random.rand(20, 5)
    I = np.sort(np.random.randint(20, size=(100,)))
    J = np.random.randint(20, size=(100,))

    return A, R, X, I, J


def test_opsax(valid_arguments):
    A, R, X, I, J = valid_arguments

    reference = ref_opsax(A, R, X, I, J)
    actual = opsax(A, R, X, I, J)
    assert np.allclose(reference, actual)


def test_opsax_no_neighbors(valid_arguments):
    A, R, X, I, J = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    I[I == 1] = 2

    reference = ref_opsax(A, R, X, I, J)
    actual = opsax(A, R, X, I, J)
    assert np.allclose(reference, actual)


def test_opsax_wrong_type(valid_arguments):
    A, R, X, I, J = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(TypeError, match="Wrong dtype for A in opsax: got int32"):
        opsax(A, R, X, I, J)


def test_opsax_wrong_number_of_dimensions(valid_arguments):
    A, R, X, I, J = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError, match="A must be a 2D array in opsax, got a 3D array"
    ):
        opsax(A, R, X, I, J)


def test_opsax_size_mismatch(valid_arguments):
    A, R, X, I, J = valid_arguments
    I = I[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and I along dimension 0 must match in opsax",
    ):
        opsax(A, R, X, I, J)


def test_opsax_out_of_bounds(valid_arguments):
    A, R, X, I, J = valid_arguments
    I[0] = X.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array I in operation opsax contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        opsax(A, R, X, I, J)
