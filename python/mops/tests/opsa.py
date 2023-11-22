import numpy as np
import pytest

import mops
from mops import reference_implementations as ref

np.random.seed(0xDEADBEEF)

np.random.seed(0xDEADBEEF)


def test_opsa():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)

    indices = np.sort(np.random.randint(10, size=(100,)))

    reference = ref.outer_product_scatter_add(A, B, indices, np.max(indices) + 1)
    actual = mops.outer_product_scatter_add(A, B, indices, np.max(indices) + 1)
    assert np.allclose(reference, actual)


def test_opsa_no_neighbors():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)

    indices = np.sort(np.random.randint(10, size=(100,)))
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices[indices == 1] = 2

    reference = ref.outer_product_scatter_add(A, B, indices, np.max(indices) + 1)
    actual = mops.outer_product_scatter_add(A, B, indices, np.max(indices) + 1)
    assert np.allclose(reference, actual)

    # just checking that the jvp runs
    grad_A, grad_B = mops.outer_product_scatter_add_vjp(
        np.ones_like(actual),
        A,
        B,
        indices,
        compute_grad_A=True,
    )

    assert grad_A.shape == A.shape
    assert grad_B is None

    grad_A, grad_B = mops.outer_product_scatter_add_vjp(
        np.ones_like(actual),
        A,
        B,
        indices,
        compute_grad_B=True,
    )

    assert grad_A is None
    assert grad_B.shape == B.shape


def test_opsa_wrong_type():
    message = (
        "`A` must be a 2D array in outer_product_scatter_add, got a 1D array instead"
    )
    with pytest.raises(ValueError, match=message):
        mops.outer_product_scatter_add(np.array([1]), 2, 3, 4)
