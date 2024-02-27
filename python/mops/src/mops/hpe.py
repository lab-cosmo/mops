import numpy as np

from . import _dispatch
from .checks import _check_hpe
from .dispatch_operation import dispatch_operation
from .utils import mops_tensor


def homogeneous_polynomial_evaluation(A, C, indices_A):
    _check_hpe(A, C, indices_A)

    A = _dispatch.make_contiguous(A)
    C = _dispatch.make_contiguous(C)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_A = indices_A.astype(np.int32)

    O = _dispatch.empty_like((A.shape[0],), A)

    function = dispatch_operation(
        "homogeneous_polynomial_evaluation",
        A,
    )

    function(
        mops_tensor(O),
        mops_tensor(A),
        mops_tensor(C),
        mops_tensor(indices_A),
    )

    return O
