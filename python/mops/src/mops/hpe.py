import numpy as np


from .checks import _check_hpe
from .utils import numpy_to_mops_tensor
from . import _dispatch
from .dispatch_operation import dispatch_operation


def homogeneous_polynomial_evaluation(A, C, indices_A):
    _check_hpe(A, C, indices_A)

    A = _dispatch.make_contiguous(A)
    C = _dispatch.make_contiguous(C)
    indices_A = _dispatch.make_contiguous(indices_A)
    indices_A = indices_A.astype(np.int32)

    O = _dispatch.empty_like((A.shape[0],), A)

    function = dispatch_operation(
        A,
        "homogeneous_polynomial_evaluation",
    )

    function(
        numpy_to_mops_tensor(O),
        numpy_to_mops_tensor(A),
        numpy_to_mops_tensor(C),
        numpy_to_mops_tensor(indices_A),
    )

    return O
