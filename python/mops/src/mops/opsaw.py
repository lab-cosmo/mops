import numpy as np

from . import _dispatch
from .checks import _check_opsaw, _check_opsaw_vjp
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
