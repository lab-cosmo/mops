import numpy as np

from ._c_lib import _get_library
from .utils import numpy_to_mops_tensor


def outer_product_scatter_add(A, B, indices):
    return _outer_product_scatter_add_numpy(A, B, indices)


def _outer_product_scatter_add_numpy(A, B, indices):
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    indices = np.ascontiguousarray(indices)

    if A.dtype != B.dtype:
        raise TypeError("A and B must have the same dtype")

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise TypeError("A and B must be 2-dimensional arrays")

    if not np.can_cast(indices, np.int32, "same_kind"):
        raise TypeError("indices must be an array of integers")

    indices = indices.astype(np.int32)

    if len(indices.shape) != 1:
        raise TypeError("indices must be 1-dimensional arrays")

    if A.shape[0] != B.shape[0] or A.shape[0] != indices.shape[0]:
        raise TypeError(
            "A, B and indices must have the same number of elements on the "
            "first dimension"
        )

    output = np.zeros((np.max(indices) + 1, A.shape[1] * B.shape[1]), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_outer_product_scatter_add_f32
    elif A.dtype == np.float64:
        function = lib.mops_outer_product_scatter_add_f64
    else:
        raise TypeError("only float32 and float64 are supported")

    function(
        numpy_to_mops_tensor(output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(indices),
    )

    return output.reshape(-1, A.shape[1], B.shape[1])
