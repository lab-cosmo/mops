import numpy as np
import pytest
from mops.reference_implementations import sparse_accumulation_of_products as ref_sap

import mops
from mops import sparse_accumulation_of_products as sap

np.random.seed(0xDEADBEEF)


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 6)
    C = np.random.rand(30)
    P_A = np.random.randint(20, size=(30,))
    P_B = np.random.randint(6, size=(30,))
    n_O = 35
    P_O = np.sort(np.random.randint(n_O, size=(30,)))
    return A, B, C, P_A, P_B, P_O, n_O


def test_sap(valid_arguments):
    A, B, C, P_A, P_B, P_O, n_O = valid_arguments

    reference = ref_sap(A, B, C, P_A, P_B, P_O, n_O)
    actual = sap(A, B, C, P_A, P_B, P_O, n_O)
    assert np.allclose(reference, actual)


def test_sap_wrong_type(valid_arguments):
    A, B, C, P_A, P_B, P_O, n_O = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(TypeError, match="Wrong dtype for A in sap: got int32"):
        sap(A, B, C, P_A, P_B, P_O, n_O)


def test_sap_wrong_number_of_dimensions(valid_arguments):
    A, B, C, P_A, P_B, P_O, n_O = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(ValueError, match="A must be a 2D array in sap, got a 3D array"):
        sap(A, B, C, P_A, P_B, P_O, n_O)


def test_sap_size_mismatch(valid_arguments):
    A, B, C, P_A, P_B, P_O, n_O = valid_arguments
    C = C[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of C along "
        "dimension 0 and P_A along dimension 0 must match in sap",
    ):
        sap(A, B, C, P_A, P_B, P_O, n_O)


def test_sap_out_of_bounds(valid_arguments):
    A, B, C, P_A, P_B, P_O, n_O = valid_arguments
    P_A[0] = A.shape[1]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array P_A in operation sap contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        sap(A, B, C, P_A, P_B, P_O, n_O)
