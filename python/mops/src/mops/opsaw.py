import numpy as np

from . import _dispatch
from .checks import _check_opsaw
from .dispatch_operation import dispatch_operation
from .utils import mops_tensor


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
