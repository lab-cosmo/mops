#include "mops/capi.hpp"

#include "mops/opsa.hpp"
#include "mops/opsa.h"

static size_t checked_cast(int64_t value) {
    if (value < 0 || static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error(
            "integer value '" + std::to_string(value) + "' does not fit in size_t"
        );
    }
    return static_cast<size_t>(value);
}


extern "C" int mops_outer_product_scatter_add_f32(
    mops_tensor_2d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add<float>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {indices_output.data, {checked_cast(indices_output.shape[0])}}
        );
    );
}


extern "C" int mops_outer_product_scatter_add_f64(
    mops_tensor_2d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add<double>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {indices_output.data, {checked_cast(indices_output.shape[0])}}
        );
    );
}


extern "C" int mops_cuda_outer_product_scatter_add_f32(
    mops_tensor_2d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add<float>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {indices_output.data, {checked_cast(indices_output.shape[0])}}
        );
    );
}


extern "C" int mops_cuda_outer_product_scatter_add_f64(
    mops_tensor_2d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add<double>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {indices_output.data, {checked_cast(indices_output.shape[0])}}
        );
    );
}
