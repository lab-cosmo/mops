import ctypes


class mops_tensor_2d_f32_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.ARRAY(ctypes.c_int64, 2)),
    ]


class mops_tensor_2d_f64_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("shape", ctypes.ARRAY(ctypes.c_int64, 2)),
    ]


class mops_tensor_1d_i32_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int32)),
        ("shape", ctypes.ARRAY(ctypes.c_int64, 1)),
    ]


def setup_functions(lib):
    from .status import _check_status

    lib.mops_get_last_error_message.argtypes = []
    lib.mops_get_last_error_message.restype = ctypes.c_char_p

    # outer_product_scatter_add
    lib.mops_outer_product_scatter_add_f32.argtypes = [
        mops_tensor_2d_f32_t,
        mops_tensor_2d_f32_t,
        mops_tensor_2d_f32_t,
        mops_tensor_1d_i32_t,
    ]
    lib.mops_outer_product_scatter_add_f32.restype = _check_status

    lib.mops_outer_product_scatter_add_f64.argtypes = [
        mops_tensor_2d_f64_t,
        mops_tensor_2d_f64_t,
        mops_tensor_2d_f64_t,
        mops_tensor_1d_i32_t,
    ]
    lib.mops_outer_product_scatter_add_f64.restype = _check_status

    lib.mops_cuda_outer_product_scatter_add_f32.argtypes = [
        mops_tensor_2d_f32_t,
        mops_tensor_2d_f32_t,
        mops_tensor_2d_f32_t,
        mops_tensor_1d_i32_t,
    ]
    lib.mops_cuda_outer_product_scatter_add_f32.restype = _check_status

    lib.mops_cuda_outer_product_scatter_add_f64.argtypes = [
        mops_tensor_2d_f64_t,
        mops_tensor_2d_f64_t,
        mops_tensor_2d_f64_t,
        mops_tensor_1d_i32_t,
    ]
    lib.mops_cuda_outer_product_scatter_add_f64.restype = _check_status
