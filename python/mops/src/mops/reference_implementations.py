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


def opsax(A, R, X, I, J, n_O):
    check_opsax(A, R, X, I, J, n_O)


def sasax(C, A, R, X, I, J, M_A, M_X, M_O, n_O1, n_O2):
    check_sasax(C, A, R, X, I, J, M_A, M_X, M_O, n_O1, n_O2)
