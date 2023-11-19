import numpy as np

from ._c_lib import _get_library
from .checks import check_opsa
from .utils import numpy_to_mops_tensor


def outer_product_scatter_add(A, B, P, n_O):
    check_opsa(A, B, P, n_O)

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)
    P = P.astype(np.int32)

    output = np.empty((n_O, A.shape[1], B.shape[1]), dtype=A.dtype)  # TODO: 3D arrays

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
