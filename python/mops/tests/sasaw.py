import numpy as np
import pytest
from mops.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasax,
)

from mops import sparse_accumulation_scatter_add_with_weights as sasax

np.random.seed(0xDEADBEEF)


def test_sasax():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 200)
    W = np.random.rand(25, 13, 200)
    C = np.random.rand(50)
    output_size_1 = 25
    indices_output_1 = np.random.randint(25, size=(100,))
    indices_W_1 = np.random.randint(25, size=(100,))
    output_size_2 = 15
    indices_A = np.random.randint(20, size=(50,))
    indices_W_2 = np.random.randint(13, size=(50,))
    indices_output_2 = np.random.randint(output_size_2, size=(50,))

    reference = ref_sasax(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size_1,
        output_size_2,
    )
    actual = sasax(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size_1,
        output_size_2,
    )
    print(reference)
    print(actual)
    assert np.allclose(reference, actual)


def test_sasax_wrong_type():
    with pytest.raises(ValueError):
        sasax(np.array([1]), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
