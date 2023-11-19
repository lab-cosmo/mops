import numpy as np
import pytest
from mops.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasax,
)

import mops
from mops import sparse_accumulation_scatter_add_with_weights as sasax

np.random.seed(0xDEADBEEF)


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 20)
    R = np.random.rand(100, 200)
    X = np.random.rand(25, 13, 200)
    C = np.random.rand(50)
    I = np.sort(np.random.randint(25, size=(100,)))
    J = np.random.randint(25, size=(100,))
    n_O = 15
    M_1 = np.random.randint(20, size=(50,))
    M_2 = np.random.randint(13, size=(50,))
    M_3 = np.random.randint(n_O, size=(50,))

    return A, R, X, C, I, J, M_1, M_2, M_3, n_O


def test_sasax(valid_arguments):
    A, R, X, C, I, J, M_1, M_2, M_3, n_O = valid_arguments

    reference = ref_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)
    actual = sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)
    assert np.allclose(reference, actual)


def test_sasax_no_neighbors(valid_arguments):
    A, R, X, C, I, J, M_1, M_2, M_3, n_O = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    I[I == 1] = 2

    reference = ref_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)
    actual = sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)
    assert np.allclose(reference, actual)


def test_sasax_wrong_type(valid_arguments):
    A, R, X, C, I, J, M_1, M_2, M_3, n_O = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(TypeError, match="Wrong dtype for A in sasax: got int32"):
        sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)


def test_sasax_wrong_number_of_dimensions(valid_arguments):
    A, R, X, C, I, J, M_1, M_2, M_3, n_O = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError, match="A must be a 2D array in sasax, got a 3D array"
    ):
        sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)


def test_sasax_size_mismatch(valid_arguments):
    A, R, X, C, I, J, M_1, M_2, M_3, n_O = valid_arguments
    I = I[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and I along dimension 0 must match in sasax",
    ):
        sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)


def test_sasax_out_of_bounds(valid_arguments):
    A, R, X, C, I, J, M_1, M_2, M_3, n_O = valid_arguments
    I[0] = X.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array I in operation sasax contains elements up to 25; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 24",
    ):
        sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)
