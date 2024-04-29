import numpy as np

from . import _dispatch
from .checks import _check_opsa, _check_opsa_vjp, _check_opsa_vjp_vjp
from .dispatch_operation import dispatch_operation
from .utils import mops_tensor, null_mops_tensor_like


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
        mops_tensor(output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(indices_output),
    )

    return output


def outer_product_scatter_add_vjp(
    grad_output,
    A,
    B,
    indices_output,
    compute_grad_A=True,
    compute_grad_B=True,
):
    _check_opsa_vjp(grad_output, A, B, indices_output, grad_output.shape[0])

    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    indices_output = _dispatch.make_contiguous(indices_output)
    indices_output = indices_output.astype(np.int32)

    if compute_grad_A:
        grad_A = _dispatch.empty_like(A.shape, A)
        mops_grad_A = mops_tensor(grad_A)
    else:
        grad_A = None
        mops_grad_A = null_mops_tensor_like(A)

    if compute_grad_B:
        grad_B = _dispatch.empty_like(B.shape, B)
        mops_grad_B = mops_tensor(grad_B)
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
        mops_tensor(grad_output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(indices_output),
    )

    return grad_A, grad_B


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

    grad_grad_A = _dispatch.make_contiguous(grad_grad_A)
    grad_grad_B = _dispatch.make_contiguous(grad_grad_B)
    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    indices_output = _dispatch.make_contiguous(indices_output)
    indices_output = indices_output.astype(np.int32)

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.empty_like(
            (grad_output.shape[0], A.shape[1], B.shape[1]), grad_output
        )
        mops_grad_grad_output = mops_tensor(grad_grad_output)
    else:
        grad_grad_output = None
        mops_grad_grad_output = null_mops_tensor_like(grad_output)

    if compute_grad_A_2:
        grad_A_2 = _dispatch.empty_like(A.shape, A)
        mops_grad_A_2 = mops_tensor(grad_A_2)
    else:
        grad_A_2 = None
        mops_grad_A_2 = null_mops_tensor_like(A)

    if compute_grad_B_2:
        grad_B_2 = _dispatch.empty_like(B.shape, B)
        mops_grad_B_2 = mops_tensor(grad_B_2)
    else:
        grad_B_2 = None
        mops_grad_B_2 = null_mops_tensor_like(B)

    function = dispatch_operation(
        "outer_product_scatter_add_vjp_vjp",
        A,
    )

    function(
        mops_grad_grad_output,
        mops_grad_A_2,
        mops_grad_B_2,
        (
            mops_tensor(grad_grad_A)
            if grad_grad_A is not None
            else null_mops_tensor_like(A)
        ),
        (
            mops_tensor(grad_grad_B)
            if grad_grad_B is not None
            else null_mops_tensor_like(B)
        ),
        mops_tensor(grad_output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(indices_output),
    )

    return grad_grad_output, grad_A_2, grad_B_2
