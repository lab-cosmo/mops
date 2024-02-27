import numpy as np

from .dispatch_operation import dispatch_operation
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

    function = dispatch_operation(
        A,
        "sparse_accumulation_of_products",
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
