import ctypes

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


def get_ctypes_pointer(array):
    if array.dtype == np.float32:
        if isinstance(array, np.ndarray):
            return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        elif isinstance(array, cupy_ndarray):
            return ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_float))
        else:
            raise TypeError(
                f"Only numpy and cupy arrays are supported, found {type(array)}"
            )
    elif array.dtype == np.float64:
        if isinstance(array, np.ndarray):
            return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        elif isinstance(array, cupy_ndarray):
            return ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double))
        else:
            raise TypeError(
                f"Only numpy and cupy arrays are supported, found {type(array)}"
            )
    elif array.dtype == np.int32:
        if isinstance(array, np.ndarray):
            return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        elif isinstance(array, cupy_ndarray):
            return ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_int32))
        else:
            raise TypeError(
                f"Only numpy and cupy arrays are supported, found {type(array)}"
            )
    else:
        raise TypeError(
            f"Only int32, float32 and float64 dtypes are supported, found {array.dtype}"
        )
