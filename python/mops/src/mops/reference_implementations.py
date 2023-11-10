import numpy as np
from checks import check_hpe, check_sap, check_opsa, check_opsax, check_sasax


def hpe(C, A, P):
    check_hpe(C, A, P)

    O = np.zeros((A.shape[0],), dtype=A.dtype)
    J = P.shape[0]
    K = P.shape[1]
    for j in range(J):
        temp = C[j]
        for k in range(K):
            temp *= A[:, P[j, k]]
        O[:] += temp


def sap(C, A, B, P_A, P_B, P_O, n_O):
    check_sap(C, A, B, P_A, P_B, P_O, n_O)

    O = np.zeros((A.shape[0], n_O), dtype=A.dtype)
    K = C.shape[0]
    for k in range(K):
        O[:, P_O[k]] += C[k] * A[:, P_A[k]] * A[:, P_B[k]]


def opsa(A, B, P, n_O):
    check_opsa(A, B, P, n_O)

    O = np.zeros((n_O, A.shape[1], B.shape[1]), dtype=A.dtype)
    J = P.shape[0]
    for j in range(J):
        O[P[j], :, :] += A[j, :, None] * B[j, None, :]


def opsax(A, R, X, I, J, n_O):
    check_opsax(A, R, X, I, J, n_O)

    O = np.zeros((n_O, A.shape[1], R.shape[1]), dtype=A.dtype)
    E = I.shape[0]
    for e in range(E):
        O[I[e], :, :] += A[e, :, None] * R[e, None, :] * X[J[e], None, :]


def sasax(C, A, R, X, I, J, M_1, M_2, M_3, n_O1, n_O2):
    check_sasax(C, A, R, X, I, J, M_1, M_2, M_3, n_O1, n_O2)

    O = np.zeros((n_O1, n_O2, A.shape[1]), dtype=A.dtype)
    E = E = I.shape[0]
    N = C.shape[0]
    for e in range(E):
        for n in range(N):
            O[I[e], M_3[n], :] += R[e, :] * C[n] * A[e, M_1[n]] * X[J[e], M_2[n], :]
