import numpy as np

try:
    import cupy as cp
    from cupy import ndarray as cupy_ndarray

except ImportError:

    class cupy_ndarray:
        pass


def zeros_like(shape, array):
    if isinstance(array, np.ndarray):
        return np.zeros(shape, dtype=array.dtype)
    elif isinstance(array, cupy_ndarray):
        return cp.zeros(shape, dtype=array.dtype)
    else:
        raise TypeError(
            f"Only numpy and cupy arrays are supported, found {type(array)}"
        )


def empty_like(shape, array):
    if isinstance(array, np.ndarray):
        return np.empty(shape, dtype=array.dtype)
    elif isinstance(array, cupy_ndarray):
        return cp.empty(shape, dtype=array.dtype)
    else:
        raise TypeError(
            f"Only numpy and cupy arrays are supported, found {type(array)}"
        )


def make_contiguous(array):
    if isinstance(array, np.ndarray):
        return np.ascontiguousarray(array)
    elif isinstance(array, cupy_ndarray):
        return cp.ascontiguousarray(array)
    else:
        raise TypeError(
            f"Only numpy and cupy arrays are supported, found {type(array)}"
        )


def is_array(variable):
    return isinstance(variable, np.ndarray) or isinstance(variable, cupy_ndarray)


def is_scalar(variable):
    return not is_array(variable)


def get_device(array):
    if isinstance(array, np.ndarray):
        return "cpu"
    elif isinstance(array, cupy_ndarray):
        return "cuda"
    else:
        raise TypeError(
            f"Only numpy and cupy arrays are supported, found {type(array)}"
        )
