import numpy as np
import pytest
from mops.reference_implementations import sparse_accumulation_of_products as ref_sap

from mops import sparse_accumulation_of_products as sap

np.random.seed(0xDEADBEEF)


def test_sap():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 6)
    C = np.random.rand(30)

    P_A = np.random.randint(20, size=(30,))
    P_B = np.random.randint(6, size=(30,))
    n_O = 35
    P_O = np.random.randint(n_O, size=(30,))

    reference = ref_sap(A, B, C, P_A, P_B, P_O, n_O)
    actual = sap(A, B, C, P_A, P_B, P_O, n_O)
    assert np.allclose(reference, actual)


def test_sap_wrong_type():
    with pytest.raises(ValueError):
        sap(np.array(1), 2, 3, 4, 5, 6, 7)
