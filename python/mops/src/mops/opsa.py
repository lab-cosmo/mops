import numpy as np

from ._c_lib import _get_library
from .checks import check_opsa
from .utils import numpy_to_mops_tensor


def outer_product_scatter_add(A, B, P, n_O):
    check_opsa(A, B, P, n_O)

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)

    # TODO: Include these checks in check_opsa
    if A.dtype != B.dtype:
        raise TypeError("A and B must have the same dtype")
    if len(A.shape) != 2 or len(B.shape) != 2:
        raise TypeError("A and B must be 2-dimensional arrays")
    if not np.can_cast(P, np.int32, "same_kind"):
        raise TypeError("P must be an array of integers")
    if len(P.shape) != 1:
        raise TypeError("P must be 1-dimensional arrays")
    if A.shape[0] != B.shape[0] or A.shape[0] != P.shape[0]:
        raise TypeError(
            "A, B and P must have the same number of elements on the " "first dimension"
        )

    P = P.astype(np.int32)

    output = np.empty((n_O, A.shape[1] * B.shape[1]), dtype=A.dtype)  # TODO: 3D arrays

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
        numpy_to_mops_tensor(P),
    )

    return output.reshape(-1, A.shape[1], B.shape[1])
