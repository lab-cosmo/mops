from . import _dispatch
from .checks import (
    _check_hpe,
    _check_hpe_vjp,
    _check_opsa,
    _check_opsa_vjp,
    _check_opsaw,
    _check_opsaw_vjp,
    _check_sap,
    _check_sap_vjp,
    _check_sasaw,
    _check_sasaw_vjp,
)


def homogeneous_polynomial_evaluation(A, C, indices_A):
    _check_hpe(A, C, indices_A)

    output = _dispatch.zeros_like((A.shape[0],), A)
    max_j = indices_A.shape[0]
    max_k = indices_A.shape[1]
    for j in range(max_j):
        temp = C[j]
        for k in range(max_k):
            temp = temp * A[:, indices_A[j, k]]
        output[:] += temp

    return output


def sparse_accumulation_of_products(
    A, B, C, indices_A, indices_B, indices_output, output_size
):
    _check_sap(A, B, C, indices_A, indices_B, indices_output, output_size)

    output = _dispatch.zeros_like((A.shape[0], output_size), A)
    K = C.shape[0]
    for k in range(K):
        output[:, indices_output[k]] += C[k] * A[:, indices_A[k]] * B[:, indices_B[k]]

    return output


def outer_product_scatter_add(A, B, indices_output, output_size):
    _check_opsa(A, B, indices_output, output_size)

    output = _dispatch.zeros_like((output_size, A.shape[1], B.shape[1]), A)
    J = indices_output.shape[0]
    for j in range(J):
        output[indices_output[j], :, :] += A[j, :, None] * B[j, None, :]

    return output


def outer_product_scatter_add_with_weights(A, B, W, indices_W, indices_output):
    _check_opsaw(A, B, W, indices_W, indices_output)

    output = _dispatch.zeros_like((W.shape[0], A.shape[1], B.shape[1]), A)
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
    output_size,
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
        output_size,
    )

    output = _dispatch.zeros_like((W.shape[0], output_size, B.shape[1]), B)
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


def homogeneous_polynomial_evaluation_vjp(grad_output, A, C, indices_A):
    _check_hpe_vjp(grad_output, A, C, indices_A)

    grad_A = _dispatch.zeros_like(A.shape, grad_output)
    max_j = indices_A.shape[0]
    max_k = indices_A.shape[1]
    for j in range(max_j):
        base_product = C[j] * grad_output
        for k in range(max_k):
            temp = base_product
            for l in range(max_k):
                if l == k:
                    continue
                temp = temp * A[:, indices_A[j, l]]
            grad_A[:, indices_A[j, k]] += temp

    return grad_A


def sparse_accumulation_of_products_vjp(
    grad_output, A, B, C, indices_A, indices_B, indices_output
):
    _check_sap_vjp(
        grad_output, A, B, C, indices_A, indices_B, indices_output, grad_output.shape[1]
    )

    grad_A = _dispatch.zeros_like(A.shape, grad_output)
    grad_B = _dispatch.zeros_like(B.shape, grad_output)
    K = C.shape[0]
    for k in range(K):
        grad_A[:, indices_A[k]] += (
            grad_output[:, indices_output[k]] * C[k] * B[:, indices_B[k]]
        )
        grad_B[:, indices_B[k]] += (
            grad_output[:, indices_output[k]] * C[k] * A[:, indices_A[k]]
        )

    return grad_A, grad_B


def outer_product_scatter_add_vjp(grad_output, A, B, indices_output):
    _check_opsa_vjp(grad_output, A, B, indices_output, grad_output.shape[0])

    grad_A = _dispatch.zeros_like(A.shape, A)
    grad_B = _dispatch.zeros_like(B.shape, B)
    J = indices_output.shape[0]
    for j in range(J):
        grad_A[j, :] += (grad_output[indices_output[j], :, :] * B[j, None, :]).sum(
            axis=1
        )
        grad_B[j, :] += (grad_output[indices_output[j], :, :] * A[j, :, None]).sum(
            axis=0
        )

    return grad_A, grad_B


def outer_product_scatter_add_with_weights_vjp(
    grad_output, A, B, W, indices_W, indices_output
):
    _check_opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)

    grad_A = _dispatch.zeros_like(A.shape, A)
    grad_B = _dispatch.zeros_like(B.shape, B)
    grad_W = _dispatch.zeros_like(W.shape, W)
    max_e = indices_output.shape[0]
    for e in range(max_e):
        grad_A[e, :] += (
            grad_output[indices_output[e], :, :]
            * B[e, None, :]
            * W[indices_W[e], None, :]
        ).sum(axis=1)
        grad_B[e, :] += (
            grad_output[indices_output[e], :, :]
            * A[e, :, None]
            * W[indices_W[e], None, :]
        ).sum(axis=0)
        grad_W[indices_W[e], :] += (
            grad_output[indices_output[e], :, :] * A[e, :, None] * B[e, None, :]
        ).sum(axis=0)

    return grad_A, grad_B, grad_W


def sparse_accumulation_scatter_add_with_weights_vjp(
    grad_output,
    A,
    B,
    C,
    W,
    indices_A,
    indices_W_1,
    indices_W_2,
    indices_output_1,
    indices_output_2,
):
    _check_sasaw_vjp(
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        grad_output.shape[1],
    )

    grad_A = _dispatch.zeros_like(A.shape, A)
    grad_B = _dispatch.zeros_like(B.shape, B)
    grad_W = _dispatch.zeros_like(W.shape, W)
    E = indices_output_1.shape[0]
    N = C.shape[0]
    for e in range(E):
        for n in range(N):
            grad_A[e, indices_A[n]] += (
                grad_output[indices_output_1[e], indices_output_2[n], :]
                * B[e, :]
                * C[n]
                * W[indices_W_1[e], indices_W_2[n], :]
            ).sum()
            grad_B[e, :] += (
                grad_output[indices_output_1[e], indices_output_2[n], :]
                * A[e, indices_A[n]]
                * C[n]
                * W[indices_W_1[e], indices_W_2[n], :]
            )
            grad_W[indices_W_1[e], indices_W_2[n], :] += (
                grad_output[indices_output_1[e], indices_output_2[n], :]
                * A[e, indices_A[n]]
                * B[e, :]
                * C[n]
            )

    return grad_A, grad_B, grad_W
