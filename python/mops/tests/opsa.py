import numpy as np

import mops


def reference_implementation(A, B, indices):
    assert A.shape[0] == B.shape[0]
    output = np.zeros((np.max(indices) + 1, A.shape[1], B.shape[1]))

    for a, b, i in zip(A, B, indices):
        output[i] += np.tensordot(a, b, axes=0)

    return output


def test_opsa_numpy():
    np.random.seed(0xDEADBEEF)

    A = np.random.rand(100, 20)
    B = np.random.rand(100, 5)

    indices = np.sort(np.random.randint(10, size=(100,)))
    # substitute all 1s by 2s so as to test the no-neighbor case
    indices[indices == 1] = 2

    reference = reference_implementation(A, B, indices)
    actual = mops.outer_product_scatter_add(A, B, indices)
    assert np.allclose(reference, actual)
