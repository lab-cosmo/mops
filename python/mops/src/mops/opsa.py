import numpy as np

from ._c_lib import _get_library
from .checks import _check_opsa
from .utils import null_mops_tensor_like, numpy_to_mops_tensor


def outer_product_scatter_add(A, B, indices_output, output_size):
    _check_opsa(A, B, indices_output, output_size)
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    indices_output = np.ascontiguousarray(indices_output)
    indices_output = indices_output.astype(np.int32)

    output = np.empty((output_size, A.shape[1], B.shape[1]), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_outer_product_scatter_add_f32
    elif A.dtype == np.float64:
        function = lib.mops_outer_product_scatter_add_f64
    else:
        raise TypeError(
            "Unsupported dtype detected. Only float32 and float64 are supported"
        )

    function(
        numpy_to_mops_tensor(output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(indices_output),
    )

    return output.reshape((output_size, A.shape[1], B.shape[1]))


def outer_product_scatter_add_vjp(
    grad_output,
    A,
    B,
    indices_output,
    compute_grad_A=False,
    compute_grad_B=False,
):
    grad_output = np.ascontiguousarray(grad_output)
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    indices_output = np.ascontiguousarray(indices_output)

    if A.dtype != B.dtype or A.dtype != grad_output.dtype:
        raise TypeError("A, B and grad_output must have the same dtype")

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise TypeError("A and B must be 2-dimensional arrays")

    if len(grad_output.shape) != 3:
        raise TypeError("grad_output must be 3-dimensional arrays")

    if not np.can_cast(indices_output, np.int32, "same_kind"):
        raise TypeError("indices_output must be an array of integers")

    indices_output = indices_output.astype(np.int32)

    if len(indices_output.shape) != 1:
        raise TypeError("indices_output must be 1-dimensional arrays")

    if A.shape[0] != B.shape[0] or A.shape[0] != indices_output.shape[0]:
        raise TypeError(
            "A, B and indices_output must have the same number of elements on the "
            "first dimension"
        )

    if compute_grad_A:
        grad_A = np.empty_like(A)
        mops_grad_A = numpy_to_mops_tensor(grad_A)
    else:
        grad_A = None
        mops_grad_A = null_mops_tensor_like(A)

    if compute_grad_B:
        grad_B = np.empty_like(B)
        mops_grad_B = numpy_to_mops_tensor(grad_B)
    else:
        grad_B = None
        mops_grad_B = null_mops_tensor_like(B)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_outer_product_scatter_add_vjp_f32
    elif A.dtype == np.float64:
        function = lib.mops_outer_product_scatter_add_vjp_f64
    else:
        raise TypeError(
            "Unsupported dtype detected. Only float32 and float64 are supported"
        )

    function(
        mops_grad_A,
        mops_grad_B,
        numpy_to_mops_tensor(grad_output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(indices_output),
    )

    return grad_A, grad_B
