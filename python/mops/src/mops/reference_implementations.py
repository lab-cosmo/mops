import numpy as np

from .checks import check_hpe, check_opsa, check_opsax, check_sap, check_sasax


def homogeneous_polynomial_evaluation(A, C, P):
    check_hpe(A, C, P)

    O = np.zeros((A.shape[0],), dtype=A.dtype)
    J = P.shape[0]
    K = P.shape[1]
    for j in range(J):
        temp = C[j]
        for k in range(K):
            temp *= A[:, P[j, k]]
        O[:] += temp

    return O


def sparse_accumulation_of_products(A, B, C, P_A, P_B, P_O, n_O):
    check_sap(A, B, C, P_A, P_B, P_O, n_O)

    O = np.zeros((A.shape[0], n_O), dtype=A.dtype)
    K = C.shape[0]
    for k in range(K):
        O[:, P_O[k]] += C[k] * A[:, P_A[k]] * B[:, P_B[k]]

    return O


def outer_product_scatter_add(A, B, P, n_O):
    check_opsa(A, B, P, n_O)

    O = np.zeros((n_O, A.shape[1], B.shape[1]), dtype=A.dtype)
    J = P.shape[0]
    for j in range(J):
        O[P[j], :, :] += A[j, :, None] * B[j, None, :]

    return O


def outer_product_scatter_add_with_weights(A, R, X, I, J, n_O):
    check_opsax(A, R, X, I, J, n_O)

    O = np.zeros((n_O, A.shape[1], R.shape[1]), dtype=A.dtype)
    E = I.shape[0]
    for e in range(E):
        O[I[e], :, :] += A[e, :, None] * R[e, None, :] * X[J[e], None, :]

    return O


def sparse_accumulation_scatter_add_with_weights(
    A, R, X, C, I, J, M_1, M_2, M_3, n_O1, n_O2
):
    check_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O1, n_O2)

    O = np.zeros((n_O1, n_O2, X.shape[2]), dtype=X.dtype)
    E = E = I.shape[0]
    N = C.shape[0]
    for e in range(E):
        for n in range(N):
            O[I[e], M_3[n], :] += R[e, :] * C[n] * A[e, M_1[n]] * X[J[e], M_2[n], :]

    return O
