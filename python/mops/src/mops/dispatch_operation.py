import numpy as np

from . import _dispatch
from ._c_lib import _get_library


def dispatch_operation(op_name, array) -> callable:
    # chooses the correct function to dispatch to based on the name of the operation,
    # the type of the array (numpy, cupy) and the array dtype

    if array.dtype == np.float32:
        dtype = "f32"
    elif array.dtype == np.float64:
        dtype = "f64"
    else:
        raise TypeError(
            "Unsupported dtype detected. Only float32 and float64 are supported"
        )

    device = _dispatch.get_device(array)

    lib = _get_library()

    function_name = "mops_"
    if device == "cuda":
        function_name += "cuda_"
    function_name += f"{op_name}_{dtype}"
    function = getattr(lib, function_name)

    return function
