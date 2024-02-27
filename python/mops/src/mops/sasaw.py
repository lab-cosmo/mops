import numpy as np

from . import _dispatch
from .checks import _check_sasaw
from .dispatch_operation import dispatch_operation
from .utils import numpy_to_mops_tensor


def sparse_accumulation_scatter_add_with_weights(
    A,
    B,
    C,
    W,
    indices_A,
    indices_W_1,
    indices_W_2,
    indices_output_1,
    indices_output_2,
    output_size,
):
    _check_sasaw(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )

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

    output = _dispatch.empty_like((W.shape[0], output_size, B.shape[1]), A)

    function = dispatch_operation(
        "sparse_accumulation_scatter_add_with_weights",
        A,
    )

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
