import numpy as np
np.random.seed(0xDEADBEEF)

import pytest

import mops
from mops.reference_implementations import opsa as ref_opsa
from mops import opsa


def test_opsa_numpy():

    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)

    indices = np.sort(np.random.randint(10, size=(100,)))
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices[indices == 1] = 2

    reference = ref_opsa(A, B, indices, np.max(indices)+1)
    actual = opsa(A, B, indices, np.max(indices)+1)
    assert np.allclose(reference, actual)


def test_opsa_wrong_type():

    with pytest.raises(ValueError):
        opsa(np.array([1]), 2, 3, 4)
