import numpy as np
import pytest
from mops.reference_implementations import homogeneous_polynomial_evaluation as ref_hpe

from mops import homogeneous_polynomial_evaluation as hpe

np.random.seed(0xDEADBEEF)


def test_hpe():
    A = np.random.rand(100, 20)
    C = np.random.rand(200)
    indices_A = np.random.randint(20, size=(200, 4))

    reference = ref_hpe(A, C, indices_A)
    actual = hpe(A, C, indices_A)
    assert np.allclose(reference, actual)


def test_hpe_wrong_type():
    with pytest.raises(ValueError):
        hpe(np.array(1), 2, 3)
