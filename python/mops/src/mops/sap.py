import numpy as np

from ._c_lib import _get_library
from .checks import _check_sap
from .utils import numpy_to_mops_tensor


def sparse_accumulation_of_products(
    A, B, C, indices_A, indices_B, indices_output, output_size
):
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    C = np.ascontiguousarray(C)
    indices_A = np.ascontiguousarray(indices_A)
    indices_B = np.ascontiguousarray(indices_B)
    indices_output = np.ascontiguousarray(indices_output)

    _check_sap(A, B, C, indices_A, indices_B, indices_output, output_size)

    indices_A = indices_A.astype(np.int32)
    indices_B = indices_B.astype(np.int32)
    indices_output = indices_output.astype(np.int32)

    output = np.empty((A.shape[0], output_size), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_sparse_accumulation_of_products_f32
    elif A.dtype == np.float64:
        function = lib.mops_sparse_accumulation_of_products_f64
    else:
        raise TypeError(
            "Unsupported dtype detected. Only float32 and float64 are supported"
        )

    function(
        numpy_to_mops_tensor(output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(C),
        numpy_to_mops_tensor(indices_A),
        numpy_to_mops_tensor(indices_B),
        numpy_to_mops_tensor(indices_output),
    )

    return output
