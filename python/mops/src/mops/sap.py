import numpy as np

from . import _dispatch
from .checks import _check_sap
from .dispatch_operation import dispatch_operation
from .utils import numpy_to_mops_tensor


def sparse_accumulation_of_products(
    A, B, C, indices_A, indices_B, indices_output, output_size
):
    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    C = _dispatch.make_contiguous(C)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_B = _dispatch.make_contiguous(indices_B)
    indices_output = _dispatch.make_contiguous(indices_output)

    _check_sap(A, B, C, indices_A, indices_B, indices_output, output_size)

    indices_A = indices_A.astype(np.int32)
    indices_B = indices_B.astype(np.int32)
    indices_output = indices_output.astype(np.int32)

    output = _dispatch.empty_like((A.shape[0], output_size), A)

    function = dispatch_operation(
        "sparse_accumulation_of_products",
        A,
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
