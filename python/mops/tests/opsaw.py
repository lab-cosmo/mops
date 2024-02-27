import numpy as np
import pytest
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsaw,
)

import mops
from mops import outer_product_scatter_add_with_weights as opsaw

np.random.seed(0xDEADBEEF)


try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 10)
    B = np.random.rand(100, 5)
    W = np.random.rand(20, 5)
    indices_W = np.random.randint(20, size=(100,))
    indices_output = np.random.randint(20, size=(100,))

    return A, B, W, indices_W, indices_output


def test_opsaw(valid_arguments):
    A, B, W, indices_W, indices_output = valid_arguments

    reference = ref_opsaw(A, B, W, indices_W, indices_output)
    actual = opsaw(A, B, W, indices_W, indices_output)
    assert np.allclose(reference, actual)


def test_opsaw_no_neighbors(valid_arguments):
    A, B, W, indices_W, indices_output = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output[indices_output == 1] = 2

    reference = ref_opsaw(A, B, W, indices_W, indices_output)
    actual = opsaw(A, B, W, indices_W, indices_output)
    assert np.allclose(reference, actual)


def test_opsaw_wrong_type(valid_arguments):
    A, B, W, indices_W, indices_output = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError,
        match="Wrong dtype for A in "
        "outer_product_scatter_add_with_weights: got int32",
    ):
        opsaw(A, B, W, indices_W, indices_output)


def test_opsaw_wrong_number_of_dimensions(valid_arguments):
    A, B, W, indices_W, indices_output = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "outer_product_scatter_add_with_weights, got a 3D array instead",
    ):
        opsaw(A, B, W, indices_W, indices_output)


def test_opsaw_size_mismatch(valid_arguments):
    A, B, W, indices_W, indices_output = valid_arguments
    indices_output = indices_output[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and indices_output along dimension 0 must match in opsaw",
    ):
        opsaw(A, B, W, indices_W, indices_output)


def test_opsaw_out_of_bounds(valid_arguments):
    A, B, W, indices_W, indices_output = valid_arguments
    indices_output[0] = W.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_output in operation opsaw "
        "contains elements up to 20; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 19",
    ):
        opsaw(A, B, W, indices_W, indices_output)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_opsaw_cupy(valid_arguments):
    A, B, W, indices_W, indices_output = valid_arguments
    A = cp.array(A)
    B = cp.array(B)
    W = cp.array(W)
    indices_W = cp.array(indices_W)
    indices_output = cp.array(indices_output)

    reference = ref_opsaw(  # noqa: F841
        A.get(), B.get(), W.get(), indices_W.get(), indices_output.get()
    )
    with pytest.raises(
        mops.status.MopsError, match="CUDA implementation does not exist yet"
    ):
        actual = opsaw(A, B, W, indices_W, indices_output)  # noqa: F841
