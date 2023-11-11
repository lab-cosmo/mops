import numpy as np

from ._c_lib import _get_library
from .checks import check_sasax
from .utils import numpy_to_mops_tensor


def sparse_accumulation_scatter_add_with_weights(A, R, X, C, I, J, M_1, M_2, M_3, n_O):
    check_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O)

    A = np.ascontiguousarray(A)
    R = np.ascontiguousarray(R)
    X = np.ascontiguousarray(X)
    C = np.ascontiguousarray(C)
    I = np.ascontiguousarray(I)
    J = np.ascontiguousarray(J)
    M_1 = np.ascontiguousarray(M_1)
    M_2 = np.ascontiguousarray(M_2)
    M_3 = np.ascontiguousarray(M_3)
    I = I.astype(np.int32)
    J = J.astype(np.int32)
    M_1 = M_1.astype(np.int32)
    M_2 = M_2.astype(np.int32)
    M_3 = M_3.astype(np.int32)

    O = np.zeros((X.shape[0], n_O, X.shape[2]), dtype=X.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_sparse_accumulation_scatter_add_with_weights_f32
    elif A.dtype == np.float64:
        function = lib.mops_sparse_accumulation_scatter_add_with_weights_f64
    else:
        raise TypeError(
            "Unsupported dtype detected. Only float32 and float64 are supported"
        )

    function(
        numpy_to_mops_tensor(O),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(R),
        numpy_to_mops_tensor(X),
        numpy_to_mops_tensor(C),
        numpy_to_mops_tensor(I),
        numpy_to_mops_tensor(J),
        numpy_to_mops_tensor(M_1),
        numpy_to_mops_tensor(M_2),
        numpy_to_mops_tensor(M_3),
    )

    return O
