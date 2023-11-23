import numpy as np

from ._c_lib import _get_library
from .utils import numpy_to_mops_tensor
from .checks import check_sasax


def sparse_accumulation_scatter_add_with_weights(A, B, C, W, indices_A, indices_W_1, indices_W_2, indices_output_1, indices_output_2, output_size_1, output_size_2):
    check_sasax(A, B, C, W, indices_A, indices_W_1, indices_W_2, indices_output_1, indices_output_2, output_size_1, output_size_2)

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    C = np.ascontiguousarray(C)
    W = np.ascontiguousarray(W)
    indices_A = np.ascontiguousarray(indices_A)
    indices_W_1 = np.ascontiguousarray(indices_W_1)
    indices_W_2 = np.ascontiguousarray(indices_W_2)
    indices_output_1 = np.ascontiguousarray(indices_output_1)
    indices_output_2 = np.ascontiguousarray(indices_output_2)
    indices_A = indices_A.astype(np.int32)
    indices_W_1 = indices_W_1.astype(np.int32)
    indices_W_2 = indices_W_2.astype(np.int32)
    indices_output_1 = indices_output_1.astype(np.int32)
    indices_output_2 = indices_output_2.astype(np.int32)

    output = np.zeros((output_size_1, output_size_2, A.shape[1]), dtype=A.dtype)

    lib = _get_library()

    if A.dtype == np.float32:
        function = lib.mops_sparse_accumulation_scatter_add_with_weights_f32
    elif A.dtype == np.float64:
        function = lib.mops_sparse_accumulation_scatter_add_with_weights_f64
    else:
        raise TypeError("Unsupported dtype detected. outputnly float32 and float64 are supported")

    function(
        numpy_to_mops_tensor(output),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(B),
        numpy_to_mops_tensor(C),
        numpy_to_mops_tensor(W),
        numpy_to_mops_tensor(indices_A),
        numpy_to_mops_tensor(indices_W_1),
        numpy_to_mops_tensor(indices_W_2),
        numpy_to_mops_tensor(indices_output_1),
        numpy_to_mops_tensor(indices_output_2),
    )

    return output
