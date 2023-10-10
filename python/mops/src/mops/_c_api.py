import ctypes


def setup_functions(lib):
    from .status import _check_status

    lib.mops_get_last_error_message.argtypes = []
    lib.mops_get_last_error_message.restype = ctypes.c_char_p

    # outer_product_scatter_add
    lib.mops_outer_product_scatter_add_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
    ]
    lib.mops_outer_product_scatter_add_f32.restype = _check_status

    lib.mops_outer_product_scatter_add_f64.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
    ]
    lib.mops_outer_product_scatter_add_f64.restype = _check_status

    lib.mops_cuda_outer_product_scatter_add_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
    ]
    lib.mops_cuda_outer_product_scatter_add_f32.restype = _check_status

    lib.mops_cuda_outer_product_scatter_add_f64.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
    ]
    lib.mops_cuda_outer_product_scatter_add_f64.restype = _check_status
