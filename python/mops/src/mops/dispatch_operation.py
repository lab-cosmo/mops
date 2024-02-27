from ._c_lib import _get_library
import numpy as np
try:
    from cupy import ndarray as cupy_ndarray

except ImportError:

    class cupy_ndarray:
        pass

    # note: the cupy dtypes are defined as their numpy
    # counterparts in the cupy module, so we don't need to redefine them here


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
    
    if isinstance(array, np.ndarray):
        device = "cpu"
    elif isinstance(array, cupy_ndarray):
        device = "cuda"
    else:
        raise TypeError(f"Only numpy and cupy arrays are supported, found {type(array)}")

    lib = _get_library()

    function_name = f"mops_"
    if device == "cuda":
        function_name += "cuda_"
    function_name += f"{op_name}_{dtype}"
    function = getattr(lib, function_name)

    return function

