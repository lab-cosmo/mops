from . import _dispatch
from .checks import (
    _check_hpe,
    _check_hpe_vjp,
    _check_hpe_vjp_vjp,
    _check_opsa,
    _check_opsa_vjp,
    _check_opsa_vjp_vjp,
    _check_opsaw,
    _check_opsaw_vjp,
    _check_opsaw_vjp_vjp,
    _check_sap,
    _check_sap_vjp,
    _check_sap_vjp_vjp,
    _check_sasaw,
    _check_sasaw_vjp,
    _check_sasaw_vjp_vjp,
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


def homogeneous_polynomial_evaluation_vjp(
    grad_output, A, C, indices_A, compute_grad_A=True
):
    _check_hpe_vjp(grad_output, A, C, indices_A)

    if compute_grad_A:
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
    else:
        grad_A = None

    return grad_A


def sparse_accumulation_of_products_vjp(
    grad_output,
    A,
    B,
    C,
    indices_A,
    indices_B,
    indices_output,
    compute_grad_A=True,
    compute_grad_B=True,
):
    _check_sap_vjp(
        grad_output, A, B, C, indices_A, indices_B, indices_output, grad_output.shape[1]
    )

    if compute_grad_A:
        grad_A = _dispatch.zeros_like(A.shape, grad_output)
    else:
        grad_A = None
    if compute_grad_B:
        grad_B = _dispatch.zeros_like(B.shape, grad_output)
    else:
        grad_B = None

    K = C.shape[0]
    for k in range(K):
        if compute_grad_A:
            grad_A[:, indices_A[k]] += (
                grad_output[:, indices_output[k]] * C[k] * B[:, indices_B[k]]
            )
        if compute_grad_B:
            grad_B[:, indices_B[k]] += (
                grad_output[:, indices_output[k]] * C[k] * A[:, indices_A[k]]
            )

    return grad_A, grad_B


def outer_product_scatter_add_vjp(
    grad_output, A, B, indices_output, compute_grad_A=True, compute_grad_B=True
):
    _check_opsa_vjp(grad_output, A, B, indices_output, grad_output.shape[0])

    if compute_grad_A:
        grad_A = _dispatch.zeros_like(A.shape, A)
    else:
        grad_A = None
    if compute_grad_B:
        grad_B = _dispatch.zeros_like(B.shape, B)
    else:
        grad_B = None

    J = indices_output.shape[0]
    for j in range(J):
        if compute_grad_A:
            grad_A[j, :] += (grad_output[indices_output[j], :, :] * B[j, None, :]).sum(
                axis=1
            )
        if compute_grad_B:
            grad_B[j, :] += (grad_output[indices_output[j], :, :] * A[j, :, None]).sum(
                axis=0
            )

    return grad_A, grad_B


def outer_product_scatter_add_with_weights_vjp(
    grad_output,
    A,
    B,
    W,
    indices_W,
    indices_output,
    compute_grad_A=True,
    compute_grad_B=True,
    compute_grad_W=True,
):
    _check_opsaw_vjp(grad_output, A, B, W, indices_W, indices_output)

    if compute_grad_A:
        grad_A = _dispatch.zeros_like(A.shape, A)
    else:
        grad_A = None
    if compute_grad_B:
        grad_B = _dispatch.zeros_like(B.shape, B)
    else:
        grad_B = None
    if compute_grad_W:
        grad_W = _dispatch.zeros_like(W.shape, W)
    else:
        grad_W = None

    max_e = indices_output.shape[0]
    for e in range(max_e):
        if compute_grad_A:
            grad_A[e, :] += (
                grad_output[indices_output[e], :, :]
                * B[e, None, :]
                * W[indices_W[e], None, :]
            ).sum(axis=1)
        if compute_grad_B:
            grad_B[e, :] += (
                grad_output[indices_output[e], :, :]
                * A[e, :, None]
                * W[indices_W[e], None, :]
            ).sum(axis=0)
        if compute_grad_W:
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
    compute_grad_A=True,
    compute_grad_B=True,
    compute_grad_W=True,
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

    if compute_grad_A:
        grad_A = _dispatch.zeros_like(A.shape, A)
    else:
        grad_A = None
    if compute_grad_B:
        grad_B = _dispatch.zeros_like(B.shape, B)
    else:
        grad_B = None
    if compute_grad_W:
        grad_W = _dispatch.zeros_like(W.shape, W)
    else:
        grad_W = None

    E = indices_output_1.shape[0]
    N = C.shape[0]
    for e in range(E):
        for n in range(N):
            if compute_grad_A:
                grad_A[e, indices_A[n]] += (
                    grad_output[indices_output_1[e], indices_output_2[n], :]
                    * B[e, :]
                    * C[n]
                    * W[indices_W_1[e], indices_W_2[n], :]
                ).sum()
            if compute_grad_B:
                grad_B[e, :] += (
                    grad_output[indices_output_1[e], indices_output_2[n], :]
                    * A[e, indices_A[n]]
                    * C[n]
                    * W[indices_W_1[e], indices_W_2[n], :]
                )
            if compute_grad_W:
                grad_W[indices_W_1[e], indices_W_2[n], :] += (
                    grad_output[indices_output_1[e], indices_output_2[n], :]
                    * A[e, indices_A[n]]
                    * B[e, :]
                    * C[n]
                )

    return grad_A, grad_B, grad_W


def homogeneous_polynomial_evaluation_vjp_vjp(
    grad_grad_A,
    grad_output,
    A,
    C,
    indices_A,
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
):
    _check_hpe_vjp_vjp(
        grad_grad_A,
        grad_output,
        A,
        C,
        indices_A,
    )

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.zeros_like(grad_output.shape, grad_output)
    else:
        grad_grad_output = None

    if compute_grad_A_2:
        grad_A_2 = _dispatch.zeros_like(A.shape, A)
    else:
        grad_A_2 = None

    max_j = indices_A.shape[0]
    max_k = indices_A.shape[1]
    for j in range(max_j):
        if grad_grad_A is not None:
            if compute_grad_grad_output:
                base_product = C[j] * grad_grad_A
                for k in range(max_k):
                    temp = base_product
                    for l in range(max_k):
                        if l == k:
                            continue
                        temp = temp * A[:, indices_A[j, l]]
                    grad_grad_output += temp
            if compute_grad_A_2:
                base = C[j]
                for k in range(max_k):
                    temp = base * grad_grad_A[:, indices_A[j, k]]
                    for l in range(max_k):
                        if l == k:
                            continue
                        for m in range(max_k):
                            if m == k or m == l:
                                continue
                            temp = temp * A[:, indices_A[j, m]]
                        grad_A_2[:, indices_A[j, l]] += temp

    return grad_grad_output, grad_A_2


def outer_product_scatter_add_vjp_vjp(
    grad_grad_A,
    grad_grad_B,
    grad_output,
    A,
    B,
    indices_output,
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
    compute_grad_B_2=True,
):
    _check_opsa_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        indices_output,
        grad_output.shape[0],
    )

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.zeros_like(
            (grad_output.shape[0], A.shape[1], B.shape[1]), grad_output
        )
    else:
        grad_grad_output = None

    if compute_grad_A_2:
        grad_A_2 = _dispatch.zeros_like(A.shape, A)
    else:
        grad_A_2 = None

    if compute_grad_B_2:
        grad_B_2 = _dispatch.zeros_like(B.shape, B)
    else:
        grad_B_2 = None

    J = indices_output.shape[0]
    for j in range(J):
        if grad_grad_A is not None:
            if compute_grad_grad_output:
                grad_grad_output[indices_output[j], :, :] += (
                    grad_grad_A[j, :, None] * B[j, None, :]
                )
            if compute_grad_B_2:
                grad_B_2[j, :] += (
                    grad_grad_A[j, :, None] * grad_output[indices_output[j], :, :]
                ).sum(axis=0)

        if grad_grad_B is not None:
            if compute_grad_grad_output:
                grad_grad_output[indices_output[j], :, :] += (
                    A[j, :, None] * grad_grad_B[j, None, :]
                )
            if compute_grad_A_2:
                grad_A_2[j, :] += (
                    grad_grad_B[j, None, :] * grad_output[indices_output[j], :, :]
                ).sum(axis=1)

    return grad_grad_output, grad_A_2, grad_B_2


def sparse_accumulation_of_products_vjp_vjp(
    grad_grad_A,
    grad_grad_B,
    grad_output,
    A,
    B,
    C,
    indices_A,
    indices_B,
    indices_output,
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
    compute_grad_B_2=True,
):
    _check_sap_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        C,
        indices_A,
        indices_B,
        indices_output,
        grad_output.shape[1],
    )

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.zeros_like((A.shape[0], grad_output.shape[1]), A)
    else:
        grad_grad_output = None

    if compute_grad_A_2:
        grad_A_2 = _dispatch.zeros_like(A.shape, A)
    else:
        grad_A_2 = None

    if compute_grad_B_2:
        grad_B_2 = _dispatch.zeros_like(B.shape, B)
    else:
        grad_B_2 = None

    K = C.shape[0]
    for k in range(K):
        if grad_grad_A is not None:
            if compute_grad_grad_output:
                grad_grad_output[:, indices_output[k]] += (
                    grad_grad_A[:, indices_A[k]] * B[:, indices_B[k]] * C[k]
                )
            if compute_grad_B_2:
                grad_B_2[:, indices_B[k]] += (
                    grad_grad_A[:, indices_output[k]] * A[:, indices_A[k]] * C[k]
                )
        if grad_grad_B is not None:
            if compute_grad_grad_output:
                grad_grad_output[:, indices_output[k]] += (
                    grad_grad_B[:, indices_B[k]] * A[:, indices_A[k]] * C[k]
                )
            if compute_grad_A_2:
                grad_A_2[:, indices_A[k]] += (
                    grad_grad_B[:, indices_output[k]] * B[:, indices_B[k]] * C[k]
                )

    return grad_grad_output, grad_A_2, grad_B_2


def outer_product_scatter_add_with_weights_vjp_vjp(
    grad_grad_A,
    grad_grad_B,
    grad_grad_W,
    grad_output,
    A,
    B,
    W,
    indices_W,
    indices_output,
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
    compute_grad_B_2=True,
    compute_grad_W_2=True,
):
    _check_opsaw_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_grad_W,
        grad_output,
        A,
        B,
        W,
        indices_W,
        indices_output,
    )

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.zeros_like(
            (grad_grad_A.shape[1], A.shape[1], B.shape[1]), A
        )
    else:
        grad_grad_output = None

    if compute_grad_A_2:
        grad_A_2 = _dispatch.zeros_like(A.shape, A)
    else:
        grad_A_2 = None

    if compute_grad_B_2:
        grad_B_2 = _dispatch.zeros_like(B.shape, B)
    else:
        grad_B_2 = None

    if compute_grad_W_2:
        grad_W_2 = _dispatch.zeros_like(W.shape, W)
    else:
        grad_W_2 = None

    max_e = indices_output.shape[0]
    for e in range(max_e):
        if grad_grad_A is not None:
            if compute_grad_grad_output:
                grad_grad_output[indices_output[e], :, :] += (
                    grad_grad_A[e, :, None] * B[e, None, :] * W[indices_W[e], None, :]
                )
            if compute_grad_B_2:
                grad_B_2[e, :] += (
                    grad_grad_A[e, :, None]
                    * grad_output[indices_output[e], :, :]
                    * W[indices_W[e], None, :]
                ).sum(axis=0)
            if compute_grad_W_2:
                grad_W_2[indices_W[e], :] += (
                    grad_grad_A[e, :, None]
                    * grad_output[indices_output[e], :, :]
                    * B[e, None, :]
                ).sum(axis=0)

        if grad_grad_B is not None:
            if compute_grad_grad_output:
                grad_grad_output[indices_output[e], :, :] += (
                    A[e, :, None] * grad_grad_B[e, None, :] * W[indices_W[e], None, :]
                )
            if compute_grad_A_2:
                grad_A_2[e, :] += (
                    grad_grad_B[e, None, :]
                    * grad_output[indices_output[e], :, :]
                    * W[indices_W[e], None, :]
                ).sum(axis=1)
            if compute_grad_W_2:
                grad_W_2[indices_W[e], :] += (
                    A[e, :, None]
                    * grad_grad_B[e, None, :]
                    * grad_output[indices_output[e], :, :]
                ).sum(axis=0)

        if grad_grad_W is not None:
            if compute_grad_grad_output:
                grad_grad_output[indices_output[e], :, :] += (
                    A[e, :, None] * B[e, None, :] * grad_grad_W[e, None, :]
                )
            if compute_grad_A_2:
                grad_A_2[e, :] += (
                    grad_grad_W[e, None, :]
                    * grad_output[indices_output[e], :, :]
                    * B[e, None, :]
                ).sum(axis=1)
            if compute_grad_B_2:
                grad_B_2[e, :] += (
                    A[e, :, None]
                    * grad_grad_W[e, None, :]
                    * grad_output[indices_output[e], :, :]
                ).sum(axis=0)

    return grad_grad_output, grad_A_2, grad_B_2, grad_W_2


def sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    grad_grad_A,
    grad_grad_B,
    grad_grad_W,
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
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
    compute_grad_B_2=True,
    compute_grad_W_2=True,
):
    _check_sasaw_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_grad_W,
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

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.empty_like(
            (grad_output.shape[1], A.shape[1], B.shape[1]), grad_output
        )
    else:
        grad_grad_output = None

    if compute_grad_A_2:
        grad_A_2 = _dispatch.empty_like(A.shape, A)
    else:
        grad_A_2 = None

    if compute_grad_B_2:
        grad_B_2 = _dispatch.empty_like(B.shape, B)
    else:
        grad_B_2 = None

    if compute_grad_W_2:
        grad_W_2 = _dispatch.empty_like(W.shape, W)
    else:
        grad_W_2 = None

    E = indices_output_1.shape[0]
    N = C.shape[0]
    for e in range(E):
        for n in range(N):
            if grad_grad_A is not None:
                if compute_grad_grad_output:
                    grad_grad_output[indices_output_1[e], indices_output_2[n], :] += (
                        grad_grad_A[e, indices_A[n]]
                        * B[e, :]
                        * C[n]
                        * W[indices_W_1[e], indices_W_2[n], :]
                    )
                if compute_grad_B_2:
                    grad_B_2[e, :] += (
                        grad_grad_A[e, indices_output_1[n]]
                        * A[e, indices_A[n]]
                        * C[n]
                        * W[indices_W_1[e], indices_W_2[n], :]
                    )
                if compute_grad_W_2:
                    grad_W_2[indices_W_1[e], indices_W_2[n], :] += (
                        grad_grad_A[e, indices_output_1[n]]
                        * A[e, indices_A[n]]
                        * B[e, :]
                        * C[n]
                    )
            if grad_grad_B is not None:
                if compute_grad_grad_output:
                    grad_grad_output[indices_output_1[e], indices_output_2[n], :] += (
                        A[e, indices_A[n]]
                        * grad_grad_B[e, :]
                        * C[n]
                        * W[indices_W_1[e], indices_W_2[n], :]
                    )
                if compute_grad_A_2:
                    grad_A_2[e, indices_A[n]] += (
                        grad_grad_B[e, :]
                        * grad_output[indices_output_1[e], indices_output_2[n], :]
                        * C[n]
                        * W[indices_W_1[e], indices_W_2[n], :]
                    ).sum()
                if compute_grad_W_2:
                    grad_W_2[indices_W_1[e], indices_W_2[n], :] += (
                        A[e, indices_A[n]]
                        * grad_grad_B[e, :]
                        * grad_output[indices_output_1[e], indices_output_2[n], :]
                        * C[n]
                    )
            if grad_grad_W is not None:
                if compute_grad_grad_output:
                    grad_grad_output[indices_output_1[e], indices_output_2[n], :] += (
                        A[e, indices_A[n]] * B[e, :] * grad_grad_W[e, :]
                    )
                if compute_grad_A_2:
                    grad_A_2[e, indices_A[n]] += (
                        grad_grad_W[e, :]
                        * grad_output[indices_output_1[e], indices_output_2[n], :]
                        * B[e, :]
                        * C[n]
                    ).sum()
                if compute_grad_B_2:
                    grad_B_2[e, :] += (
                        A[e, indices_A[n]]
                        * grad_grad_W[e, :]
                        * grad_output[indices_output_1[e], indices_output_2[n], :]
                        * C[n]
                    )

    return grad_grad_output, grad_A_2, grad_B_2, grad_W_2
