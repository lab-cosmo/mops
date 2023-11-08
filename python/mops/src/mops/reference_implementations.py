import numpy as np
from checks import check_hpe, check_sap, check_opsa, check_opsax, check_sasax


def hpe(C, A, P):
    check_hpe(C, A, P)


def sap(C, A, B, P_A, P_B, P_O, n_O):
    check_sap(C, A, B, P_A, P_B, P_O, n_O)


def opsa(A, B, P):
    check_opsa(A, B, P)


def opsax(A, R, X, I, J):
    check_opsax(A, R, X, I, J)


def sasax(C, A, R, X, I, J, M_A, M_X, M_O, n_O):
    check_sasax(C, A, R, X, I, J, M_A, M_X, M_O, n_O)
