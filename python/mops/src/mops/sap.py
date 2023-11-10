import numpy as np

from ._c_lib import _get_library
from .utils import numpy_to_mops_tensor
from .checks import check_sap


def sparse_accumulation_of_products(C, A, B, P_A, P_B, P_O, n_O):
    check_sap(C, A, B, P_A, P_B, P_O, n_O)

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    P = np.ascontiguousarray(P)
    P = P.astype(np.int32)

    O = np.zeros((A.shape[0], n_O), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_sparse_accumulation_of_products_f32
    elif A.dtype == np.float64:
        function = lib.mops_sparse_accumulation_of_products_f64
    else:
        raise TypeError("Unsupported dtype detected. Only float32 and float64 are supported")

    function(
        numpy_to_mops_tensor(O),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(P),
    )

    return O
