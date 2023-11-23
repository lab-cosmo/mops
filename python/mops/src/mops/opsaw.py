import numpy as np

from ._c_lib import _get_library
from .utils import numpy_to_mops_tensor
from .checks import check_opsaw


def outer_product_scatter_add_with_weights(A, B, W, indices_w, indices_output, output_size):
    check_opsaw(A, B, W, indices_w, indices_output, output_size)

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    W = np.ascontiguousarray(W)
    indices_w = np.ascontiguousarray(indices_w)
    indices_output = np.ascontiguousarray(indices_output)
    indices_w = indices_w.astype(np.int32)
    indices_output = indices_output.astype(np.int32)

    output = np.zeros((output_size, A.shape[1], B.shape[1]), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_outer_product_scatter_add_with_weights_f32
    elif A.dtype == np.float64:
        function = lib.mops_outer_product_scatter_add_with_weights_f64
    else:
        raise TypeError("Unsupported dtype detected. outputnly float32 and float64 are supported")

    function(
        numpy_to_mops_tensor(output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(W),
        numpy_to_mops_tensor(indices_w),
        numpy_to_mops_tensor(indices_output),
    )

    return output
