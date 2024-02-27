import numpy as np

from .dispatch_operation import dispatch_operation
from .checks import _check_opsaw
from .utils import numpy_to_mops_tensor


def outer_product_scatter_add_with_weights(A, B, W, indices_w, indices_output):
    _check_opsaw(A, B, W, indices_w, indices_output)

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    W = np.ascontiguousarray(W)
    indices_w = np.ascontiguousarray(indices_w)
    indices_output = np.ascontiguousarray(indices_output)
    indices_w = indices_w.astype(np.int32)
    indices_output = indices_output.astype(np.int32)

    output = np.empty((W.shape[0], A.shape[1], B.shape[1]), dtype=A.dtype)

    function = dispatch_operation(
        A,
        "outer_product_scatter_add_with_weights",
    )

    function(
        numpy_to_mops_tensor(output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(W),
        numpy_to_mops_tensor(indices_w),
        numpy_to_mops_tensor(indices_output),
    )

    return output
