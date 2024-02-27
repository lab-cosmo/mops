import numpy as np
import pytest
from mops.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasaw,
)

import mops
from mops import sparse_accumulation_scatter_add_with_weights as sasaw

np.random.seed(0xDEADBEEF)


try:
    import cupy as cp

    cp.random.seed(0xDEADBEEF)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.fixture
def valid_arguments():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 200)
    W = np.random.rand(25, 13, 200)
    C = np.random.rand(50)
    indices_output_1 = np.random.randint(25, size=(100,))
    indices_W_1 = np.random.randint(25, size=(100,))
    output_size = 15
    indices_A = np.random.randint(20, size=(50,))
    indices_W_2 = np.random.randint(13, size=(50,))
    indices_output_2 = np.random.randint(output_size, size=(50,))

    return (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )


def test_sasaw(valid_arguments):
    (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    ) = valid_arguments

    reference = ref_sasaw(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )
    actual = sasaw(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )
    assert np.allclose(reference, actual)


def test_sasaw_no_neighbors(valid_arguments):
    (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    ) = valid_arguments
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices_output_1[indices_output_1 == 1] = 2

    reference = ref_sasaw(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )
    actual = sasaw(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )
    assert np.allclose(reference, actual)


def test_sasaw_wrong_type(valid_arguments):
    (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    ) = valid_arguments
    A = A.astype(np.int32)

    with pytest.raises(
        TypeError,
        match="Wrong dtype for A in "
        "sparse_accumulation_scatter_add_with_weights: got int32",
    ):
        sasaw(
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
            output_size,
        )


def test_sasaw_wrong_number_of_dimensions(valid_arguments):
    (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    ) = valid_arguments
    A = A[..., np.newaxis]

    with pytest.raises(
        ValueError,
        match="`A` must be a 2D array in "
        "sparse_accumulation_scatter_add_with_weights, got a 3D array instead",
    ):
        sasaw(
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
            output_size,
        )


def test_sasaw_size_mismatch(valid_arguments):
    (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    ) = valid_arguments
    indices_output_1 = indices_output_1[:5]

    with pytest.raises(
        mops.status.MopsError,
        match="Dimension mismatch: the sizes of A along "
        "dimension 0 and indices_output_1 along dimension 0 must match in sasaw",
    ):
        sasaw(
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
            output_size,
        )


def test_sasaw_out_of_bounds(valid_arguments):
    (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    ) = valid_arguments
    indices_output_1[0] = W.shape[0]

    with pytest.raises(
        mops.status.MopsError,
        match="Index array indices_output_1 in operation "
        "sasaw contains elements up to 25; "
        "this would cause out-of-bounds accesses. With the provided "
        "parameters, it can only contain elements up to 24",
    ):
        sasaw(
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
            output_size,
        )


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")
def test_sasaw_cupy(valid_arguments):
    (
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    ) = valid_arguments
    A = cp.array(A)
    B = cp.array(B)
    C = cp.array(C)
    W = cp.array(W)
    indices_A = cp.array(indices_A)
    indices_W_1 = cp.array(indices_W_1)
    indices_W_2 = cp.array(indices_W_2)
    indices_output_1 = cp.array(indices_output_1)
    indices_output_2 = cp.array(indices_output_2)

    reference = ref_sasaw(  # noqa: F841
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )
    with pytest.raises(
        mops.status.MopsError, match="CUDA implementation does not exist yet"
    ):
        actual = sasaw(  # noqa: F841
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
            output_size,
        )
