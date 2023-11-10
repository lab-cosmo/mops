import numpy as np

from ._c_lib import _get_library
from .utils import numpy_to_mops_tensor
from .checks import check_hpe


def homogeneous_polynomial_evaluation(A, C, P):
    check_hpe(A, C, P)

    A = np.ascontiguousarray(A)
    C = np.ascontiguousarray(C)
    P = np.ascontiguousarray(P)

    O = np.zeros((A.shape[0],), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_homogeneous_polynomial_evaluation_f32
    elif A.dtype == np.float64:
        function = lib.mops_homogeneous_polynomial_evaluation_f64
    else:
        raise TypeError("Unsupported dtype detected. Only float32 and float64 are supported")

    function(
        numpy_to_mops_tensor(O),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(C),
        numpy_to_mops_tensor(P),
    )

    return O
