import numpy as np

from ._c_lib import _get_library
from .checks import check_opsax
from .utils import numpy_to_mops_tensor


def outer_product_scatter_add_with_weights(A, R, X, I, J, n_O):
    check_opsax(A, R, X, I, J, n_O)

    A = np.ascontiguousarray(A)
    R = np.ascontiguousarray(R)
    X = np.ascontiguousarray(X)
    I = np.ascontiguousarray(I)
    J = np.ascontiguousarray(J)
    I = I.astype(np.int32)
    J = J.astype(np.int32)

    O = np.zeros((n_O, A.shape[1], R.shape[1]), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_outer_product_scatter_add_with_weights_f32
    elif A.dtype == np.float64:
        function = lib.mops_outer_product_scatter_add_with_weights_f64
    else:
        raise TypeError(
            "Unsupported dtype detected. Only float32 and float64 are supported"
        )

    function(
        numpy_to_mops_tensor(O),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(R),
        numpy_to_mops_tensor(X),
        numpy_to_mops_tensor(I),
        numpy_to_mops_tensor(J),
    )

    return O
