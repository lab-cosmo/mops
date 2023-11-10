import numpy as np
np.random.seed(0xDEADBEEF)

import pytest

from mops.reference_implementations import sparse_accumulation_of_products as ref_sap
from mops import sparse_accumulation_of_products as sap


def test_sap():

    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)

    indices = np.sort(np.random.randint(10, size=(100,)))

    reference = ref_sap(A, B, indices, np.max(indices)+1)
    actual = sap(A, B, indices, np.max(indices)+1)
    assert np.allclose(reference, actual)


def test_sap_wrong_type():

    with pytest.raises(ValueError):
        sap(np.array([1]), 2, 3, 4, 5, 6, 7)
