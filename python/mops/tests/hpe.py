import numpy as np
np.random.seed(0xDEADBEEF)

import pytest

from mops.reference_implementations import homogeneous_polynomial_evaluation as ref_hpe
from mops import homogeneous_polynomial_evaluation as hpe


def test_hpe():
    
    A = np.random.rand(100, 20)
    C = np.random.rand(200)
    P = np.random.randint(20, size=(200, 4))

    reference = ref_hpe(A, C, P)
    actual = hpe(A, C, P)
    assert np.allclose(reference, actual)


def test_hpe_wrong_type():

    with pytest.raises(ValueError):
        hpe(np.array(1), 2, 3)
