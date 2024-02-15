import numpy as np

# For each operation, we only check the correctness of types and number of dimensions.
# Size consistency checks will be performed in the C++ backend.


def _check_number_of_dimensions(array, expected, operation, name):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"`{name}` must be an array in {operation}, got {type(array)}")

    if len(array.shape) != expected:
        raise ValueError(
            f"`{name}` must be a {expected}D array in {operation}, "
            f"got a {len(array.shape)}D array instead"
        )


def _check_scalar(variable, operation, name):
    if isinstance(variable, np.ndarray):
        raise TypeError(f"{name} must be a scalar in {operation}, found a `np.ndarray`")


def _check_array_dtype(array, expected_dtype, operation, name):
    if not np.issubdtype(array.dtype, expected_dtype):
        raise TypeError(f"Wrong dtype for {name} in {operation}: got {array.dtype}")


def _check_scalar_dtype(scalar, expected_dtype, operation, name):
    if not np.issubdtype(type(scalar), expected_dtype):
        raise TypeError(f"Wrong dtype for {name} in {operation}: got {type(scalar)}")


def _check_hpe(A, C, indices_A):
    function = "homogeneous_polynomial_evaluation"

    # Check dimensions
    _check_number_of_dimensions(A, 2, function, "A")
    _check_number_of_dimensions(C, 1, function, "C")
    _check_number_of_dimensions(indices_A, 2, function, "indices_A")

    # Check types
    _check_array_dtype(A, np.floating, function, "A")
    _check_array_dtype(C, np.floating, function, "C")
    _check_array_dtype(indices_A, np.integer, function, "indices_A")


def _check_sap(A, B, C, indices_A, indices_B, indices_output, output_size):
    function = "sparse_accumulation_of_products"

    # Check dimensions
    _check_number_of_dimensions(A, 2, function, "A")
    _check_number_of_dimensions(B, 2, function, "B")
    _check_number_of_dimensions(C, 1, function, "C")
    _check_number_of_dimensions(indices_A, 1, function, "indices_A")
    _check_number_of_dimensions(indices_B, 1, function, "indices_B")
    _check_number_of_dimensions(indices_output, 1, function, "indices_output")
    _check_scalar(output_size, function, "output_size")

    # Check types
    _check_array_dtype(A, np.floating, function, "A")
    _check_array_dtype(B, np.floating, function, "B")
    _check_array_dtype(C, np.floating, function, "C")
    _check_array_dtype(indices_A, np.integer, function, "indices_A")
    _check_array_dtype(indices_B, np.integer, function, "indices_B")
    _check_array_dtype(indices_output, np.integer, function, "indices_output")
    _check_scalar_dtype(output_size, np.integer, function, "output_size")


def _check_opsa(A, B, indices_output, output_size):
    function = "outer_product_scatter_add"

    # Check dimensions
    _check_number_of_dimensions(A, 2, function, "A")
    _check_number_of_dimensions(B, 2, function, "B")
    _check_number_of_dimensions(indices_output, 1, function, indices_output)
    _check_scalar(output_size, function, output_size)

    # Check types
    _check_array_dtype(A, np.floating, function, "A")
    _check_array_dtype(B, np.floating, function, "B")
    _check_array_dtype(indices_output, np.integer, function, indices_output)
    _check_scalar_dtype(output_size, np.integer, function, output_size)


def _check_opsaw(A, B, W, indices_W, indices_output):
    function = "outer_product_scatter_add_with_weights"

    # Check dimensions
    _check_number_of_dimensions(A, 2, function, "A")
    _check_number_of_dimensions(B, 2, function, "B")
    _check_number_of_dimensions(W, 2, function, "W")
    _check_number_of_dimensions(indices_W, 1, function, "indices_W")
    _check_number_of_dimensions(indices_output, 1, function, "indices_output")

    # Check types
    _check_array_dtype(A, np.floating, function, "A")
    _check_array_dtype(B, np.floating, function, "B")
    _check_array_dtype(W, np.floating, function, "W")
    _check_array_dtype(indices_W, np.integer, function, "indices_W")
    _check_array_dtype(indices_output, np.integer, function, "indices_output")


def _check_sasaw(
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
    function = "sparse_accumulation_scatter_add_with_weights"

    # Check dimensions
    _check_number_of_dimensions(A, 2, function, "A")
    _check_number_of_dimensions(B, 2, function, "B")
    _check_number_of_dimensions(C, 1, function, "C")
    _check_number_of_dimensions(W, 3, function, "W")
    _check_number_of_dimensions(indices_A, 1, function, "indices_A")
    _check_number_of_dimensions(indices_W_1, 1, function, "indices_W_1")
    _check_number_of_dimensions(indices_W_2, 1, function, "indices_W_2")
    _check_number_of_dimensions(indices_output_1, 1, function, "indices_output_1")
    _check_number_of_dimensions(indices_output_2, 1, function, "indices_output_2")
    _check_scalar(output_size, function, "output_size")

    # Check types
    _check_array_dtype(A, np.floating, function, "A")
    _check_array_dtype(B, np.floating, function, "R")
    _check_array_dtype(C, np.floating, function, "X")
    _check_array_dtype(W, np.floating, function, "X")
    _check_array_dtype(indices_A, np.integer, function, "indices_A")
    _check_array_dtype(indices_W_1, np.integer, function, "indices_W_1")
    _check_array_dtype(indices_W_2, np.integer, function, "indices_W_2")
    _check_array_dtype(indices_output_1, np.integer, function, "indices_output_1")
    _check_array_dtype(indices_output_2, np.integer, function, "indices_output_2")
    _check_scalar_dtype(output_size, np.integer, function, "output_size")
