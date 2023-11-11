import numpy as np
import pytest
from mops.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasax,
)

from mops import sparse_accumulation_scatter_add_with_weights as sasax

np.random.seed(0xDEADBEEF)


def test_sasax():
    A = np.random.rand(100, 20)
    R = np.random.rand(100, 200)
    X = np.random.rand(25, 13, 200)
    C = np.random.rand(50)
    I = np.random.randint(25, size=(100,))
    J = np.random.randint(25, size=(100,))
    n_O = 15
    M_1 = np.random.randint(20, size=(50,))
    M_2 = np.random.randint(13, size=(50,))
    M_3 = np.random.randint(n_O, size=(50,))

    reference = ref_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)
    actual = sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)
    print(reference)
    print(actual)
    assert np.allclose(reference, actual)


def test_sasax_wrong_type():
    with pytest.raises(ValueError):
        sasax(np.array([1]), 2, 3, 4, 5, 6, 7, 8, 9, 10)
