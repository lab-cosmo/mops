import numpy as np

from . import _dispatch
from .checks import _check_hpe, _check_hpe_vjp, _check_hpe_vjp_vjp
from .dispatch_operation import dispatch_operation
from .utils import mops_tensor, null_mops_tensor_like


def homogeneous_polynomial_evaluation(A, C, indices_A):
    _check_hpe(A, C, indices_A)

    A = _dispatch.make_contiguous(A)
    C = _dispatch.make_contiguous(C)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_A = indices_A.astype(np.int32)

    output = _dispatch.empty_like((A.shape[0],), A)

    function = dispatch_operation(
        "homogeneous_polynomial_evaluation",
        A,
    )

    function(
        mops_tensor(output),
        mops_tensor(A),
        mops_tensor(C),
        mops_tensor(indices_A),
    )

    return output


def homogeneous_polynomial_evaluation_vjp(
    grad_output, A, C, indices_A, compute_grad_A=True
):
    _check_hpe_vjp(grad_output, A, C, indices_A)

    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    C = _dispatch.make_contiguous(C)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_A = indices_A.astype(np.int32)

    if compute_grad_A:
        grad_A = _dispatch.empty_like(A.shape, A)
        mops_grad_A = mops_tensor(grad_A)
    else:
        grad_A = None
        mops_grad_A = null_mops_tensor_like(A)

    function = dispatch_operation(
        "homogeneous_polynomial_evaluation_vjp",
        A,
    )

    function(
        mops_grad_A,
        mops_tensor(grad_output),
        mops_tensor(A),
        mops_tensor(C),
        mops_tensor(indices_A),
    )

    return grad_A


def homogeneous_polynomial_evaluation_vjp_vjp(
    grad_grad_A,
    grad_output,
    A,
    C,
    indices_A,
    compute_grad_grad_output=True,
    compute_grad_A_2=True,
):
    _check_hpe_vjp_vjp(
        grad_grad_A,
        grad_output,
        A,
        C,
        indices_A,
    )

    grad_output = _dispatch.make_contiguous(grad_output)
    A = _dispatch.make_contiguous(A)
    C = _dispatch.make_contiguous(C)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_A = indices_A.astype(np.int32)

    if compute_grad_grad_output:
        grad_grad_output = _dispatch.empty_like(grad_output.shape, grad_output)
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

    function = dispatch_operation(
        "homogeneous_polynomial_evaluation_vjp_vjp",
        A,
    )

    function(
        mops_grad_A_2,
        mops_grad_grad_output,
        (
            mops_tensor(grad_grad_A)
            if grad_grad_A is not None
            else null_mops_tensor_like(A)
        ),
        mops_tensor(grad_output),
        mops_tensor(A),
        mops_tensor(C),
        mops_tensor(indices_A),
    )

    return grad_grad_output, grad_A_2
