import numpy as np

from . import _dispatch
from .checks import _check_sasaw
from .dispatch_operation import dispatch_operation
from .utils import mops_tensor


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

    A = _dispatch.make_contiguous(A)
    B = _dispatch.make_contiguous(B)
    C = _dispatch.make_contiguous(C)
    W = _dispatch.make_contiguous(W)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_W_1 = _dispatch.make_contiguous(indices_W_1)
    indices_W_2 = _dispatch.make_contiguous(indices_W_2)
    indices_output_1 = _dispatch.make_contiguous(indices_output_1)
    indices_output_2 = _dispatch.make_contiguous(indices_output_2)
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
        mops_tensor(output),
        mops_tensor(A),
        mops_tensor(B),
        mops_tensor(C),
        mops_tensor(W),
        mops_tensor(indices_A),
        mops_tensor(indices_W_1),
        mops_tensor(indices_W_2),
        mops_tensor(indices_output_1),
        mops_tensor(indices_output_2),
    )

    return output
