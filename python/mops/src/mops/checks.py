import numpy as np

from . import _dispatch

# For each operation, we only check the correctness of types and number of dimensions.
# Size consistency checks will be performed in the C++ backend.


def _check_number_of_dimensions(array, expected, operation, name):
    if not _dispatch.is_array(array):
        raise TypeError(f"`{name}` must be an array in {operation}, got {type(array)}")

    if len(array.shape) != expected:
        raise ValueError(
            f"`{name}` must be a {expected}D array in {operation}, "
            f"got a {len(array.shape)}D array instead"
        )


def _check_scalar(variable, operation, name):
    if not _dispatch.is_scalar(variable):
        raise TypeError(
            f"`{name}` must be a scalar in {operation}, got {type(variable)}"
        )


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


def _check_hpe_vjp(grad_output, A, C, indices_A):
    function = "homogeneous_polynomial_evaluation_vjp"

    _check_number_of_dimensions(grad_output, 1, function, "grad_output")
    _check_array_dtype(grad_output, np.floating, function, "grad_output")

    _check_hpe(A, C, indices_A)


def _check_sap_vjp(
    grad_output, A, B, C, indices_A, indices_B, indices_output, output_size
):
    function = "sparse_accumulation_of_products_vjp"

    _check_number_of_dimensions(grad_output, 2, function, "grad_output")
    _check_array_dtype(grad_output, np.floating, function, "grad_output")

    _check_sap(A, B, C, indices_A, indices_B, indices_output, output_size)


def _check_opsa_vjp(grad_output, A, B, indices_output, output_size):
    function = "outer_product_scatter_add_vjp"

    _check_number_of_dimensions(grad_output, 3, function, "grad_output")
    _check_array_dtype(grad_output, np.floating, function, "grad_output")

    _check_opsa(A, B, indices_output, output_size)


def _check_opsaw_vjp(grad_output, A, B, W, indices_W, indices_output):
    function = "outer_product_scatter_add_with_weights_vjp"

    _check_number_of_dimensions(grad_output, 3, function, "grad_output")
    _check_array_dtype(grad_output, np.floating, function, "grad_output")

    _check_opsaw(A, B, W, indices_W, indices_output)


def _check_sasaw_vjp(
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
    output_size,
):
    function = "sparse_accumulation_scatter_add_with_weights"

    _check_number_of_dimensions(grad_output, 3, function, "grad_output")
    _check_array_dtype(grad_output, np.floating, function, "grad_output")

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


def _check_hpe_vjp_vjp(
    grad_grad_A,
    grad_output,
    A,
    C,
    indices_A,
):
    function = "homogeneous_polynomial_evaluation_vjp_vjp"

    if grad_grad_A is not None:
        _check_number_of_dimensions(grad_grad_A, 2, function, "grad_grad_A")

    if grad_grad_A is not None:
        _check_array_dtype(grad_grad_A, np.floating, function, "grad_grad_A")

    _check_hpe_vjp(grad_output, A, C, indices_A)


def _check_sap_vjp_vjp(
    grad_grad_A,
    grad_grad_B,
    grad_output,
    A,
    B,
    C,
    indices_A,
    indices_B,
    indices_output,
    output_size,
):
    function = "sparse_accumulation_of_products_vjp_vjp"

    if grad_grad_A is not None:
        _check_number_of_dimensions(grad_grad_A, 2, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_number_of_dimensions(grad_grad_B, 2, function, "grad_grad_B")

    if grad_grad_A is not None:
        _check_array_dtype(grad_grad_A, np.floating, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_array_dtype(grad_grad_B, np.floating, function, "grad_grad_B")

    _check_sap_vjp(
        grad_output, A, B, C, indices_A, indices_B, indices_output, output_size
    )


def _check_opsa_vjp_vjp(
    grad_grad_A, grad_grad_B, grad_output, A, B, indices_output, output_size
):
    function = "outer_product_scatter_add_vjp_vjp"

    if grad_grad_A is not None:
        _check_number_of_dimensions(grad_grad_A, 2, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_number_of_dimensions(grad_grad_B, 2, function, "grad_grad_B")

    if grad_grad_A is not None:
        _check_array_dtype(grad_grad_A, np.floating, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_array_dtype(grad_grad_B, np.floating, function, "grad_grad_B")

    _check_opsa_vjp(A, B, indices_output, grad_output, output_size)


def _check_opsaw_vjp_vjp(
    grad_grad_A,
    grad_grad_B,
    grad_grad_W,
    grad_output,
    A,
    B,
    W,
    indices_W,
    indices_output,
):
    function = "outer_product_scatter_add_with_weights_vjp_vjp"

    if grad_grad_A is not None:
        _check_number_of_dimensions(grad_grad_A, 2, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_number_of_dimensions(grad_grad_B, 2, function, "grad_grad_B")
    if grad_grad_W is not None:
        _check_number_of_dimensions(grad_grad_W, 2, function, "grad_grad_W")

    if grad_grad_A is not None:
        _check_array_dtype(grad_grad_A, np.floating, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_array_dtype(grad_grad_B, np.floating, function, "grad_grad_B")
    if grad_grad_W is not None:
        _check_array_dtype(grad_grad_W, np.floating, function, "grad_grad_W")

    _check_opsaw_vjp(A, B, W, indices_W, indices_output, grad_output)


def _check_sasaw_vjp_vjp(
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
    output_size,
):
    function = "sparse_accumulation_scatter_add_with_weights_vjp_vjp"

    if grad_grad_A is not None:
        _check_number_of_dimensions(grad_grad_A, 2, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_number_of_dimensions(grad_grad_B, 2, function, "grad_grad_B")
    if grad_grad_W is not None:
        _check_number_of_dimensions(grad_grad_W, 3, function, "grad_grad_W")

    if grad_grad_A is not None:
        _check_array_dtype(grad_grad_A, np.floating, function, "grad_grad_A")
    if grad_grad_B is not None:
        _check_array_dtype(grad_grad_B, np.floating, function, "grad_grad_B")
    if grad_grad_W is not None:
        _check_array_dtype(grad_grad_W, np.floating, function, "grad_grad_W")

    _check_sasaw_vjp(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        grad_output,
        output_size,
    )
