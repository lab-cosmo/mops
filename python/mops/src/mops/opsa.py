import numpy as np

from ._c_lib import _get_library
from .checks import _check_opsa
from .utils import numpy_to_mops_tensor


def outer_product_scatter_add(A, B, indices_output, output_size):
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    indices_output = np.ascontiguousarray(indices_output)

    _check_opsa(A, B, indices_output, output_size)

    # TODO: Include these checks in check_opsa
    if A.dtype != B.dtype:
        raise TypeError("A and B must have the same dtype")
    if len(A.shape) != 2 or len(B.shape) != 2:
        raise TypeError("A and B must be 2-dimensional arrays")
    if not np.can_cast(indices_output, np.int32, "same_kind"):
        raise TypeError("`indices_output` must be an array of 32-bit integers")

    if A.shape[0] != B.shape[0] or A.shape[0] != indices_output.shape[0]:
        raise TypeError(
            "A, B and indices_output must have the same number "
            "of elements along the first dimension"
        )

    indices_output = indices_output.astype(np.int32)

    output = np.zeros((output_size, A.shape[1] * B.shape[1]), dtype=A.dtype)

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
        numpy_to_mops_tensor(indices_output),
    )

    return output.reshape(-1, A.shape[1], B.shape[1])
