import numpy as np
import pytest
from mops.reference_implementations import outer_product_scatter_add as ref_opsa

from mops import outer_product_scatter_add as opsa

np.random.seed(0xDEADBEEF)


def test_opsa():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)

    indices = np.sort(np.random.randint(10, size=(100,)))

    reference = ref_opsa(A, B, indices, np.max(indices) + 1)
    actual = opsa(A, B, indices, np.max(indices) + 1)
    assert np.allclose(reference, actual)


def test_opsa_no_neighbors():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)

    indices = np.sort(np.random.randint(10, size=(100,)))
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices[indices == 1] = 2

    reference = ref_opsa(A, B, indices, np.max(indices) + 1)
    actual = opsa(A, B, indices, np.max(indices) + 1)
    assert np.allclose(reference, actual)


def test_opsa_wrong_type():
    with pytest.raises(ValueError):
        opsa(np.array([1]), 2, 3, 4)
