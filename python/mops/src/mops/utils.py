import ctypes

import numpy as np

from . import _dispatch
from ._c_api import (
    mops_tensor_1d_f32_t,
    mops_tensor_1d_f64_t,
    mops_tensor_1d_i32_t,
    mops_tensor_2d_f32_t,
    mops_tensor_2d_f64_t,
    mops_tensor_2d_i32_t,
    mops_tensor_3d_f32_t,
    mops_tensor_3d_f64_t,
)


def mops_tensor(array):
    assert _dispatch.is_array(array)

    if array.dtype == np.float32:
        if len(array.shape) == 1:
            tensor = mops_tensor_1d_f32_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            return tensor
        elif len(array.shape) == 2:
            tensor = mops_tensor_2d_f32_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            tensor.shape[1] = array.shape[1]
            return tensor
        elif len(array.shape) == 3:
            tensor = mops_tensor_3d_f32_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            tensor.shape[1] = array.shape[1]
            tensor.shape[2] = array.shape[2]
            return tensor
        else:
            raise TypeError("we can only convert 1D and 2D arrays of float32")
    elif array.dtype == np.float64:
        if len(array.shape) == 1:
            tensor = mops_tensor_1d_f64_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            return tensor
        elif len(array.shape) == 2:
            tensor = mops_tensor_2d_f64_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            tensor.shape[1] = array.shape[1]
            return tensor
        elif len(array.shape) == 3:
            tensor = mops_tensor_3d_f64_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            tensor.shape[1] = array.shape[1]
            tensor.shape[2] = array.shape[2]
            return tensor
        else:
            raise TypeError("we can only convert 1D, 2D and 3D arrays of float64")
    elif array.dtype == np.int32:
        if len(array.shape) == 1:
            tensor = mops_tensor_1d_i32_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            return tensor
        elif len(array.shape) == 2:
            tensor = mops_tensor_2d_i32_t()
            tensor.data = _dispatch.get_ctypes_pointer(array)
            tensor.shape[0] = array.shape[0]
            tensor.shape[1] = array.shape[1]
            return tensor
        else:
            raise TypeError("we can only convert 1D and 2D arrays of int32")
    else:
        raise TypeError("we can only convert arrays of int32, float32 or float64")


def null_mops_tensor_like(array):
    if array.dtype == np.float32:
        if len(array.shape) == 2:
            tensor = mops_tensor_2d_f32_t()
            tensor.data = None
            tensor.shape[0] = 0
            tensor.shape[1] = 0
            return tensor
        else:
            raise TypeError("we can only convert 2D arrays of float32")
    elif array.dtype == np.float64:
        if len(array.shape) == 2:
            tensor = mops_tensor_2d_f64_t()
            tensor.data = None
            tensor.shape[0] = 0
            tensor.shape[1] = 0
            return tensor
        else:
            raise TypeError("we can only convert 2D arrays of float64")
    elif array.dtype == np.int32:
        if len(array.shape) == 1:
            tensor = mops_tensor_1d_i32_t()
            tensor.data = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            tensor.shape[0] = array.shape[0]
            return tensor
        else:
            raise TypeError("we can only convert 1D arrays of int32")
    else:
        raise TypeError("we can only arrays of int32, float32 or float64")
