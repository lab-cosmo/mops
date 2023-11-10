import numpy as np

from ._c_lib import _get_library
from .utils import numpy_to_mops_tensor
from .checks import check_sap


def sparse_accumulation_of_products(C, A, B, P_A, P_B, P_O, n_O):
    check_sap(C, A, B, P_A, P_B, P_O, n_O)

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    C = np.ascontiguousarray(C)
    P_A = np.ascontiguousarray(P_A)
    P_B = np.ascontiguousarray(P_B)
    P_O = np.ascontiguousarray(P_O)
    P_A = P_A.astype(np.int32)
    P_B = P_B.astype(np.int32)
    P_O = P_O.astype(np.int32)

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
        numpy_to_mops_tensor(C),
        numpy_to_mops_tensor(P_A),
        numpy_to_mops_tensor(P_B),
        numpy_to_mops_tensor(P_O),
    )

    return O
