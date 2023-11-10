import numpy as np
np.random.seed(0xDEADBEEF)

import pytest

from mops.reference_implementations import sparse_accumulation_of_products as ref_sap
from mops import sparse_accumulation_of_products as sap


def test_sap():

    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)
    C = np.random.rand(25)

    P_A = np.random.randint(20, size=(25,))
    P_B = np.random.randint(5, size=(25,))
    n_O = 30
    P_O = np.random.randint(n_O, size=(25,))

    reference = ref_sap(C, A, B, P_A, P_B, P_O, n_O)
    actual = sap(C, A, B, P_A, P_B, P_O, n_O)
    print(reference)
    print(actual)
    assert np.allclose(reference, actual)


def test_sap_wrong_type():

    with pytest.raises(ValueError):
        sap(np.array(1), 2, 3, 4, 5, 6, 7)
