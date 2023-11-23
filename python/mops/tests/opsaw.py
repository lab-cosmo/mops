import numpy as np
import pytest
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsax,
)

import mops
from mops import outer_product_scatter_add_with_weights as opsax

np.random.seed(0xDEADBEEF)


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 10)
    B = np.random.rand(100, 5)
    W = np.random.rand(20, 5)
    indices_output = np.random.randint(20, size=(100,))
    indices_W = np.sort(np.random.randint(20, size=(100,)))

    return A, B, W, indices_output, indices_W


def test_opsax(valid_arguments):
    A, B, W, indices_output, indices_W = valid_arguments

    reference = ref_opsax(A, B, W, indices_output, indices_W)
    actual = opsax(A, B, W, indices_output, indices_W)
    assert np.allclose(reference, actual)


def test_opsax_no_neighbors(valid_arguments):
    A, B, W, indices_output, indices_W = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output[indices_output == 1] = 2

    reference = ref_opsax(A, B, W, indices_output, indices_W)
    actual = opsax(A, B, W, indices_output, indices_W)
    assert np.allclose(reference, actual)


def test_opsax_wrong_type(valid_arguments):
    A, B, W, indices_output, indices_W = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(TypeError, match="Wrong dtype for A in opsax: got int32"):
        opsax(A, B, W, indices_output, indices_W)


def test_opsax_wrong_number_of_dimensions(valid_arguments):
    A, B, W, indices_output, indices_W = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError, match="A must be a 2D array in opsax, got a 3D array"
    ):
        opsax(A, B, W, indices_output, indices_W)


def test_opsax_size_mismatch(valid_arguments):
    A, B, W, indices_output, indices_W = valid_arguments
    indices_output = indices_output[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and indices_output along dimension 0 must match in opsax",
    ):
        opsax(A, B, W, indices_output, indices_W)


def test_opsax_out_of_bounds(valid_arguments):
    A, B, W, indices_output, indices_W = valid_arguments
    indices_output[0] = W.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_output in operation opsax contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        opsax(A, B, W, indices_output, indices_W)
