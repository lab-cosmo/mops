import numpy as np
import pytest
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsax,
)

from mops import outer_product_scatter_add_with_weights as opsax

np.random.seed(0xDEADBEEF)


def test_opsax():
    A = np.random.rand(100, 10)
    R = np.random.rand(100, 5)
    n_O = 20
    X = np.random.rand(n_O, 5)

    I = np.random.randint(20, size=(100,))
    J = np.random.randint(20, size=(100,))

    reference = ref_opsax(A, R, X, I, J, 20)
    actual = opsax(A, R, X, I, J, 20)
    assert np.allclose(reference, actual)


def test_opsax_wrong_type():
    with pytest.raises(ValueError):
        opsax(np.array([1]), 2, 3, 4, 5, 6)
