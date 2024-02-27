import numpy as np

from . import _dispatch
from .checks import _check_opsa
from .dispatch_operation import dispatch_operation
from .utils import null_mops_tensor_like, numpy_to_mops_tensor


def outer_product_scatter_add(A, B, indices_output, output_size):
    _check_opsa(A, B, indices_output, output_size)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    indices_output = _dispatch.make_contiguous(indices_output)
    indices_output = indices_output.astype(np.int32)

    output = _dispatch.empty_like((output_size, A.shape[1], B.shape[1]), A)

    function = dispatch_operation(
        "outer_product_scatter_add",
        A,
    )

    function(
        numpy_to_mops_tensor(output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(indices_output),
    )

    return output


def outer_product_scatter_add_vjp(
    grad_output,
    A,
    B,
    indices_output,
    compute_grad_A=False,
    compute_grad_B=False,
):
    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    indices_output = _dispatch.make_contiguous(indices_output)

    if A.dtype != B.dtype or A.dtype != grad_output.dtype:
        raise TypeError("A, B and grad_output must have the same dtype")

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise TypeError("A and B must be 2-dimensional arrays")

    if len(grad_output.shape) != 3:
        raise TypeError("grad_output must be 3-dimensional arrays")

    # if not np.can_cast(indices_output, np.int32, "same_kind"):
    #     raise TypeError("indices_output must be an array of integers")

    indices_output = indices_output.astype(np.int32)

    if len(indices_output.shape) != 1:
        raise TypeError("indices_output must be 1-dimensional arrays")

    if A.shape[0] != B.shape[0] or A.shape[0] != indices_output.shape[0]:
        raise TypeError(
            "A, B and indices_output must have the same number of elements on the "
            "first dimension"
        )

    if compute_grad_A:
        grad_A = _dispatch.empty_like(A.shape, A)
        mops_grad_A = numpy_to_mops_tensor(grad_A)
    else:
        grad_A = None
        mops_grad_A = null_mops_tensor_like(A)

    if compute_grad_B:
        grad_B = _dispatch.empty_like(B.shape, B)
        mops_grad_B = numpy_to_mops_tensor(grad_B)
    else:
        grad_B = None
        mops_grad_B = null_mops_tensor_like(B)

    function = dispatch_operation(
        "outer_product_scatter_add_vjp",
        A,
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
