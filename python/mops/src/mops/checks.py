import numpy as np


# For each operation, we only check the correctness of types and number of dimensions.
# Size consistency checks will be performed in the C++ backend. 


def check_number_of_dimensions(array, expected_number_of_dimensions, operation, input):
    if len(array.shape) != expected_number_of_dimensions:
        raise ValueError(
            f"{input} must be a {expected_number_of_dimensions}D "
            f"array in {operation}, got a {len(array.shape)}D array"
        )


def check_scalar(variable, operation, input):
    if isinstance(variable, np.ndarray):
        raise TypeError(
            f"{input} must be a scalar in {operation}, found a `np.ndarray`"
        )


def check_array_dtype(array, expected_dtype, operation, input):
    if not np.issubdtype(array.dtype, expected_dtype):
        raise TypeError(f"Wrong dtype for {input} in {operation}: got {array.dtype}")


def check_scalar_dtype(scalar, expected_dtype, operation, input):
    if not np.issubdtype(type(scalar), expected_dtype):
        raise TypeError(f"Wrong dtype for {input} in {operation}: got {type(scalar)}")


def check_hpe(A, C, P):
    # Check dimensions
    check_number_of_dimensions(A, 2, "hpe", "A")
    check_number_of_dimensions(C, 1, "hpe", "C")
    check_number_of_dimensions(P, 2, "hpe", "P")

    # Check types
    check_array_dtype(A, np.floating, "hpe", "A")
    check_array_dtype(C, np.floating, "hpe", "C")
    check_array_dtype(P, np.integer, "hpe", "P")


def check_sap(A, B, C, P_A, P_B, P_O, n_O):
    # Check dimensions
    check_number_of_dimensions(A, 2, "sap", "A")
    check_number_of_dimensions(B, 2, "sap", "B")
    check_number_of_dimensions(C, 1, "sap", "C")
    check_number_of_dimensions(P_A, 1, "sap", "P_A")
    check_number_of_dimensions(P_B, 1, "sap", "P_B")
    check_number_of_dimensions(P_O, 1, "sap", "P_O")
    check_scalar(n_O, "sap", "n_O")

    # Check types
    check_array_dtype(A, np.floating, "sap", "A")
    check_array_dtype(B, np.floating, "sap", "B")
    check_array_dtype(C, np.floating, "sap", "C")
    check_array_dtype(P_A, np.integer, "sap", "P_A")
    check_array_dtype(P_B, np.integer, "sap", "P_B")
    check_array_dtype(P_O, np.integer, "sap", "P_O")
    check_scalar_dtype(n_O, np.integer, "sap", "n_O")


def check_opsa(A, B, P, n_O):
    # Check dimensions
    check_number_of_dimensions(A, 2, "opsa", "A")
    check_number_of_dimensions(B, 2, "opsa", "B")
    check_number_of_dimensions(P, 1, "opsa", "P")
    check_scalar(n_O, "opsa", "n_O")

    # Check types
    check_array_dtype(A, np.floating, "opsa", "A")
    check_array_dtype(B, np.floating, "opsa", "B")
    check_array_dtype(P, np.integer, "opsa", "P")
    check_scalar_dtype(n_O, np.integer, "opsa", "n_O")


def check_opsax(A, R, X, I, J):
    # Check dimensions
    check_number_of_dimensions(A, 2, "opsax", "A")
    check_number_of_dimensions(R, 2, "opsax", "R")
    check_number_of_dimensions(X, 2, "opsax", "X")
    check_number_of_dimensions(I, 1, "opsax", "I")
    check_number_of_dimensions(J, 1, "opsax", "J")

    # Check types
    check_array_dtype(A, np.floating, "opsax", "A")
    check_array_dtype(R, np.floating, "opsax", "R")
    check_array_dtype(X, np.floating, "opsax", "X")
    check_array_dtype(I, np.integer, "opsax", "I")
    check_array_dtype(J, np.integer, "opsax", "J")


def check_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O):
    # Check dimensions
    check_number_of_dimensions(A, 2, "sasax", "A")
    check_number_of_dimensions(R, 2, "sasax", "R")
    check_number_of_dimensions(X, 3, "sasax", "X")
    check_number_of_dimensions(C, 1, "sasax", "C")
    check_number_of_dimensions(I, 1, "sasax", "I")
    check_number_of_dimensions(J, 1, "sasax", "J")
    check_number_of_dimensions(M_1, 1, "sasax", "M_1")
    check_number_of_dimensions(M_2, 1, "sasax", "M_2")
    check_number_of_dimensions(M_3, 1, "sasax", "M_3")
    check_scalar(n_O, "sasax", n_O)

    # Check types
    check_array_dtype(A, np.floating, "sasax", "A")
    check_array_dtype(R, np.floating, "sasax", "R")
    check_array_dtype(X, np.floating, "sasax", "X")
    check_array_dtype(I, np.integer, "sasax", "I")
    check_array_dtype(J, np.integer, "sasax", "J")
    check_array_dtype(M_1, np.integer, "sasax", "M_1")
    check_array_dtype(M_2, np.integer, "sasax", "M_2")
    check_array_dtype(M_3, np.integer, "sasax", "M_3")
    check_scalar_dtype(n_O, np.integer, "sasax", "n_O")
