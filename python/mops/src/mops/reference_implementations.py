import numpy as np

from .checks import _check_hpe, _check_opsa, _check_opsaw, _check_sap, _check_sasaw


def homogeneous_polynomial_evaluation(A, C, indices_A):
    _check_hpe(A, C, indices_A)

    output = np.zeros((A.shape[0],), dtype=A.dtype)
    max_j = indices_A.shape[0]
    max_k = indices_A.shape[1]
    for j in range(max_j):
        temp = C[j]
        for k in range(max_k):
            temp *= A[:, indices_A[j, k]]
        output[:] += temp

    return output


def sparse_accumulation_of_products(
    A, B, C, indices_A, indices_B, indices_output, output_size
):
    _check_sap(A, B, C, indices_A, indices_B, indices_output, output_size)

    output = np.zeros((A.shape[0], output_size), dtype=A.dtype)
    K = C.shape[0]
    for k in range(K):
        output[:, indices_output[k]] += C[k] * A[:, indices_A[k]] * B[:, indices_B[k]]

    return output


def outer_product_scatter_add(A, B, indices_output, output_size):
    _check_opsa(A, B, indices_output, output_size)

    output = np.zeros((output_size, A.shape[1], B.shape[1]), dtype=A.dtype)
    J = indices_output.shape[0]
    for j in range(J):
        output[indices_output[j], :, :] += A[j, :, None] * B[j, None, :]

    return output


def outer_product_scatter_add_with_weights(
    A, B, W, indices_W, indices_output, output_size
):
    _check_opsaw(A, B, W, indices_W, indices_output, output_size)

    output = np.zeros((output_size, A.shape[1], B.shape[1]), dtype=A.dtype)
    max_e = indices_output.shape[0]
    for e in range(max_e):
        output[indices_output[e], :, :] += (
            A[e, :, None] * B[e, None, :] * W[indices_W[e], None, :]
        )

    return output


def sparse_accumulation_scatter_add_with_weights(
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
):
    _check_sasaw(
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

    output = np.zeros((output_size_1, output_size_2, B.shape[1]), dtype=B.dtype)
    E = indices_output_1.shape[0]
    N = C.shape[0]
    for e in range(E):
        for n in range(N):
            output[indices_output_1[e], indices_output_2[n], :] += (
                A[e, indices_A[n]]
                * B[e, :]
                * C[n]
                * W[indices_W_1[e], indices_W_2[n], :]
            )

    return output
