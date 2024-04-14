import numpy as np

from . import _dispatch
from .checks import _check_opsaw, _check_opsaw_vjp, _check_opsaw_vjp_vjp
from .dispatch_operation import dispatch_operation
from .utils import mops_tensor, null_mops_tensor_like


def outer_product_scatter_add_with_weights(A, B, W, indices_w, indices_output):
    _check_opsaw(A, B, W, indices_w, indices_output)

    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    W = _dispatch.make_contiguous(W)
    indices_w = _dispatch.make_contiguous(indices_w)
    indices_output = _dispatch.make_contiguous(indices_output)
    indices_w = indices_w.astype(np.int32)
    indices_output = indices_output.astype(np.int32)

    output = np.empty((W.shape[0], A.shape[1], B.shape[1]), dtype=A.dtype)

    function = dispatch_operation(
        "outer_product_scatter_add_with_weights",
        A,
    )

    function(
        mops_tensor(output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(W),
        mops_tensor(indices_w),
        mops_tensor(indices_output),
    )

    return output


def outer_product_scatter_add_with_weights_vjp(
    grad_output,
    A,
    B,
    W,
    indices_w,
    indices_output,
    compute_grad_A=True,
    compute_grad_B=True,
    compute_grad_W=True,
):
    _check_opsaw_vjp(grad_output, A, B, W, indices_w, indices_output)

    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    W = _dispatch.make_contiguous(W)
    indices_w = _dispatch.make_contiguous(indices_w)
    indices_output = _dispatch.make_contiguous(indices_output)

    indices_w = indices_w.astype(np.int32)
    indices_output = indices_output.astype(np.int32)

    if compute_grad_A:
        grad_A = np.empty(A.shape, dtype=A.dtype)
        mops_grad_A = mops_tensor(grad_A)
    else:
        grad_A = None
        mops_grad_A = null_mops_tensor_like(A)

    if compute_grad_B:
        grad_B = np.empty(B.shape, dtype=B.dtype)
        mops_grad_B = mops_tensor(grad_B)
    else:
        grad_B = None
        mops_grad_B = null_mops_tensor_like(B)

    if compute_grad_W:
        grad_W = np.empty(W.shape, dtype=W.dtype)
        mops_grad_W = mops_tensor(grad_W)
    else:
        grad_W = None
        mops_grad_W = null_mops_tensor_like(W)

    function = dispatch_operation(
        "outer_product_scatter_add_with_weights_vjp",
        A,
    )

    function(
        mops_grad_A,
        mops_grad_B,
        mops_grad_W,
        mops_tensor(grad_output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(W),
        mops_tensor(indices_w),
        mops_tensor(indices_output),
    )

    return grad_A, grad_B, grad_W


def outer_product_scatter_add_with_weights_vjp_vjp(
    grad_grad_A,
    grad_grad_B,
    grad_grad_W,
    grad_output,
    A,
    B,
    W,
    indices_W,
    indices_output,
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
    compute_grad_B_2=True,
    compute_grad_W_2=True,
):
    _check_opsaw_vjp_vjp(
        grad_grad_A,
        grad_grad_B,
        grad_grad_W,
        grad_output,
        A,
        B,
        W,
        indices_W,
        indices_output,
    )

    grad_grad_A = _dispatch.make_contiguous(grad_grad_A)
    grad_grad_B = _dispatch.make_contiguous(grad_grad_B)
    grad_grad_W = _dispatch.make_contiguous(grad_grad_W)
    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    W = _dispatch.make_contiguous(W)
    indices_W = _dispatch.make_contiguous(indices_W)
    indices_output = _dispatch.make_contiguous(indices_output)

    indices_W = indices_W.astype(np.int32)
    indices_output = indices_output.astype(np.int32)

    if compute_grad_grad_output:
        grad_grad_output = np.empty(
            (grad_grad_A.shape[1], A.shape[1], B.shape[1]), dtype=grad_grad_A.dtype
        )
        mops_grad_grad_output = mops_tensor(grad_grad_output)
    else:
        grad_grad_output = None
        mops_grad_grad_output = null_mops_tensor_like(grad_grad_A)

    if compute_grad_A_2:
        grad_A_2 = np.empty(A.shape, dtype=A.dtype)
        mops_grad_A_2 = mops_tensor(grad_A_2)
    else:
        grad_A_2 = None
        mops_grad_A_2 = null_mops_tensor_like(A)

    if compute_grad_B_2:
        grad_B_2 = np.empty(B.shape, dtype=B.dtype)
        mops_grad_B_2 = mops_tensor(grad_B_2)
    else:
        grad_B_2 = None
        mops_grad_B_2 = null_mops_tensor_like(B)

    if compute_grad_W_2:
        grad_W_2 = np.empty(W.shape, dtype=W.dtype)
        mops_grad_W_2 = mops_tensor(grad_W_2)
    else:
        grad_W_2 = None
        mops_grad_W_2 = null_mops_tensor_like(W)

    function = dispatch_operation(
        "outer_product_scatter_add_with_weights_vjp_vjp",
        A,
    )

    function(
        mops_grad_grad_output,
        mops_grad_A_2,
        mops_grad_B_2,
        mops_grad_W_2,
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
        (
            mops_tensor(grad_grad_W)
            if grad_grad_W is not None
            else null_mops_tensor_like(W)
        ),
        mops_tensor(grad_output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(W),
        mops_tensor(indices_W),
        mops_tensor(indices_output),
    )

    return grad_grad_output, grad_A_2, grad_B_2, grad_W_2
