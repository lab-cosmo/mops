#include "mops/capi.hpp"

#include "mops/hpe.hpp"
#include "mops/hpe.h"

static size_t checked_cast(int64_t value) {
    if (value < 0 || static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error(
            "integer value '" + std::to_string(value) + "' does not fit in size_t"
        );
    }
    return static_cast<size_t>(value);
}


extern "C" int mops_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_1d_f32_t C,
    mops_tensor_2d_i32_t indices_A
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::homogeneous_polynomial_evaluation<float>(
            {output.data, {checked_cast(output.shape[0])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {indices_A.data, {checked_cast(indices_A.shape[0]), checked_cast(indices_A.shape[1])}}
        );
    );
}


extern "C" int mops_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_1d_f64_t C,
    mops_tensor_2d_i32_t indices_A
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::homogeneous_polynomial_evaluation<double>(
            {output.data, {checked_cast(output.shape[0])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {indices_A.data, {checked_cast(indices_A.shape[0]), checked_cast(indices_A.shape[1])}}
        );
    );
}


extern "C" int mops_homogeneous_polynomial_evaluation_vjp_f32(
    mops_tensor_2d_f32_t grad_A,
    mops_tensor_1d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_1d_f32_t C,
    mops_tensor_2d_i32_t indices_A
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::homogeneous_polynomial_evaluation_vjp<float>(
            {grad_A.data, {checked_cast(grad_A.shape[0]), checked_cast(grad_A.shape[1])}},
            {grad_output.data, {checked_cast(grad_output.shape[0])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {indices_A.data, {checked_cast(indices_A.shape[0]), checked_cast(indices_A.shape[1])}}
        );
    );
}


extern "C" int mops_homogeneous_polynomial_evaluation_vjp_f64(
    mops_tensor_2d_f64_t grad_A,
    mops_tensor_1d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_1d_f64_t C,
    mops_tensor_2d_i32_t indices_A
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::homogeneous_polynomial_evaluation_vjp<double>(
            {grad_A.data, {checked_cast(grad_A.shape[0]), checked_cast(grad_A.shape[1])}},
            {grad_output.data, {checked_cast(grad_output.shape[0])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {indices_A.data, {checked_cast(indices_A.shape[0]), checked_cast(indices_A.shape[1])}}
        );
    );
}


extern "C" int mops_cuda_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_1d_f32_t C,
    mops_tensor_2d_i32_t indices_A
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::homogeneous_polynomial_evaluation<float>(
            {output.data, {checked_cast(output.shape[0])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {indices_A.data, {checked_cast(indices_A.shape[0]), checked_cast(indices_A.shape[1])}}
        );
    );
}


extern "C" int mops_cuda_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_1d_f64_t C,
    mops_tensor_2d_i32_t indices_A
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::homogeneous_polynomial_evaluation<double>(
            {output.data, {checked_cast(output.shape[0])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {indices_A.data, {checked_cast(indices_A.shape[0]), checked_cast(indices_A.shape[1])}}
        );
    );
}
