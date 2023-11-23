import numpy as np
import pytest
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsax,
)

from mops import outer_product_scatter_add_with_weights as opsax

np.random.seed(0xDEADBEEF)


def test_opsax():
    A = np.random.rand(100, 10)
    B = np.random.rand(100, 5)
    n_O = 20
    W = np.random.rand(n_O, 5)

    indices_W = np.random.randint(20, size=(100,))
    indices_output = np.random.randint(20, size=(100,))

    reference = ref_opsax(A, B, W, indices_W, indices_output, 20)
    actual = opsax(A, B, W, indices_W, indices_output, 20)
    print(reference)
    print(actual)
    assert np.allclose(reference, actual)


def test_opsax_wrong_type():
    with pytest.raises(ValueError):
        opsax(np.array([1]), 2, 3, 4, 5, 6)
