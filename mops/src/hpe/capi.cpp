#include "mops/capi.hpp"

#include "mops/hpe.h"
#include "mops/hpe.hpp"

static size_t checked_cast(int64_t value) {
    if (value < 0 ||
        static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error("integer value '" + std::to_string(value) +
                                 "' does not fit in size_t");
    }
    return static_cast<size_t>(value);
}

extern "C" int mops_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_1d_f32_t tensor_c, mops_tensor_2d_i32_t p) {
    MOPS_CATCH_EXCEPTIONS(
        mops::homogeneous_polynomial_evaluation<float>(
            {output.data, {checked_cast(output.shape[0])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_c.data, {checked_cast(tensor_c.shape[0])}},
            {p.data, {checked_cast(p.shape[0]), checked_cast(p.shape[1])}}););
}

extern "C" int mops_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_1d_f64_t tensor_c, mops_tensor_2d_i32_t p) {
    MOPS_CATCH_EXCEPTIONS(
        mops::homogeneous_polynomial_evaluation<double>(
            {output.data, {checked_cast(output.shape[0])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_c.data, {checked_cast(tensor_c.shape[0])}},
            {p.data, {checked_cast(p.shape[0]), checked_cast(p.shape[1])}}););
}

extern "C" int mops_cuda_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_1d_f32_t tensor_c, mops_tensor_2d_i32_t p) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::homogeneous_polynomial_evaluation<float>(
            {output.data, {checked_cast(output.shape[0])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_c.data, {checked_cast(tensor_c.shape[0])}},
            {p.data, {checked_cast(p.shape[0]), checked_cast(p.shape[1])}}););
}

extern "C" int mops_cuda_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_1d_f64_t tensor_c, mops_tensor_2d_i32_t p) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::homogeneous_polynomial_evaluation<double>(
            {output.data, {checked_cast(output.shape[0])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_c.data, {checked_cast(tensor_c.shape[0])}},
            {p.data, {checked_cast(p.shape[0]), checked_cast(p.shape[1])}}););
}
