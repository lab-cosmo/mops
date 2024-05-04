import numpy as np

from . import _dispatch
from .checks import _check_sasaw, _check_sasaw_vjp, _check_sasaw_vjp_vjp
from .dispatch_operation import dispatch_operation
from .utils import mops_tensor, null_mops_tensor_like


def sparse_accumulation_scatter_add_with_weights(
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

    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    C = _dispatch.make_contiguous(C)
    W = _dispatch.make_contiguous(W)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_W_1 = _dispatch.make_contiguous(indices_W_1)
    indices_W_2 = _dispatch.make_contiguous(indices_W_2)
    indices_output_1 = _dispatch.make_contiguous(indices_output_1)
    indices_output_2 = _dispatch.make_contiguous(indices_output_2)
    indices_A = indices_A.astype(np.int32)
    indices_W_1 = indices_W_1.astype(np.int32)
    indices_W_2 = indices_W_2.astype(np.int32)
    indices_output_1 = indices_output_1.astype(np.int32)
    indices_output_2 = indices_output_2.astype(np.int32)

    output = _dispatch.empty_like((W.shape[0], output_size, B.shape[1]), A)

    function = dispatch_operation(
        "sparse_accumulation_scatter_add_with_weights",
        A,
    )

    function(
        mops_tensor(output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(C),
        mops_tensor(W),
        mops_tensor(indices_A),
        mops_tensor(indices_W_1),
        mops_tensor(indices_W_2),
        mops_tensor(indices_output_1),
        mops_tensor(indices_output_2),
    )

    return output


def sparse_accumulation_scatter_add_with_weights_vjp(
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
    compute_grad_A=True,
    compute_grad_B=True,
    compute_grad_C=True,
    compute_grad_W=True,
):
    _check_sasaw_vjp(
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
        grad_output.shape[1],
    )

    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    C = _dispatch.make_contiguous(C)
    W = _dispatch.make_contiguous(W)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_W_1 = _dispatch.make_contiguous(indices_W_1)
    indices_W_2 = _dispatch.make_contiguous(indices_W_2)
    indices_output_1 = _dispatch.make_contiguous(indices_output_1)
    indices_output_2 = _dispatch.make_contiguous(indices_output_2)

    indices_A = indices_A.astype(np.int32)
    indices_W_1 = indices_W_1.astype(np.int32)
    indices_W_2 = indices_W_2.astype(np.int32)
    indices_output_1 = indices_output_1.astype(np.int32)
    indices_output_2 = indices_output_2.astype(np.int32)

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

    if compute_grad_W:
        grad_W = _dispatch.empty_like(W.shape, W)
        mops_grad_W = mops_tensor(grad_W)
    else:
        grad_W = None
        mops_grad_W = null_mops_tensor_like(W)

    function = dispatch_operation(
        "sparse_accumulation_scatter_add_with_weights_vjp",
        A,
    )

    function(
        mops_grad_A,
        mops_grad_B,
        mops_grad_W,
        mops_tensor(grad_output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(C),
        mops_tensor(W),
        mops_tensor(indices_A),
        mops_tensor(indices_W_1),
        mops_tensor(indices_W_2),
        mops_tensor(indices_output_1),
        mops_tensor(indices_output_2),
    )

    return grad_A, grad_B, grad_W


def sparse_accumulation_scatter_add_with_weights_vjp_vjp(
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
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
    compute_grad_B_2=True,
    compute_grad_W_2=True,
):
    _check_sasaw_vjp_vjp(
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
        grad_output.shape[1],
    )

    grad_grad_A = _dispatch.make_contiguous(grad_grad_A)
    grad_grad_B = _dispatch.make_contiguous(grad_grad_B)
    grad_grad_W = _dispatch.make_contiguous(grad_grad_W)
    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    C = _dispatch.make_contiguous(C)
    W = _dispatch.make_contiguous(W)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_W_1 = _dispatch.make_contiguous(indices_W_1)
    indices_W_2 = _dispatch.make_contiguous(indices_W_2)
    indices_output_1 = _dispatch.make_contiguous(indices_output_1)
    indices_output_2 = _dispatch.make_contiguous(indices_output_2)

    indices_A = indices_A.astype(np.int32)
    indices_W_1 = indices_W_1.astype(np.int32)
    indices_W_2 = indices_W_2.astype(np.int32)
    indices_output_1 = indices_output_1.astype(np.int32)
    indices_output_2 = indices_output_2.astype(np.int32)

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.empty_like(
            (grad_output.shape[1], A.shape[1], B.shape[1]), grad_output
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

    if compute_grad_W_2:
        grad_W_2 = _dispatch.empty_like(W.shape, W)
        mops_grad_W_2 = mops_tensor(grad_W_2)
    else:
        grad_W_2 = None
        mops_grad_W_2 = null_mops_tensor_like(W)

    function = dispatch_operation(
        "sparse_accumulation_scatter_add_with_weights_vjp_vjp",
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
        mops_tensor(C),
        mops_tensor(W),
        mops_tensor(indices_A),
        mops_tensor(indices_W_1),
        mops_tensor(indices_W_2),
        mops_tensor(indices_output_1),
        mops_tensor(indices_output_2),
    )

    return grad_grad_output, grad_A_2, grad_B_2, grad_W_2
