#include "mops/capi.hpp"

#include "mops/sap.hpp"
#include "mops/sap.h"

static size_t checked_cast(int64_t value) {
    if (value < 0 || static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error(
            "integer value '" + std::to_string(value) + "' does not fit in size_t"
        );
    }
    return static_cast<size_t>(value);
}


extern "C" int mops_sparse_accumulation_of_products_f32(
    mops_tensor_2d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_b,
    mops_tensor_1d_i32_t indexes
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_of_products<float>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {tensor_a.data, {checked_cast(tensor_a.shape[0]), checked_cast(tensor_a.shape[1])}},
            {tensor_b.data, {checked_cast(tensor_b.shape[0]), checked_cast(tensor_b.shape[1])}},
            {indexes.data, {checked_cast(indexes.shape[0])}}
        );
    );
}


extern "C" int mops_sparse_accumulation_of_products_f64(
    mops_tensor_2d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_b,
    mops_tensor_1d_i32_t indexes
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_of_products<double>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {tensor_a.data, {checked_cast(tensor_a.shape[0]), checked_cast(tensor_a.shape[1])}},
            {tensor_b.data, {checked_cast(tensor_b.shape[0]), checked_cast(tensor_b.shape[1])}},
            {indexes.data, {checked_cast(indexes.shape[0])}}
        );
    );
}


extern "C" int mops_cuda_sparse_accumulation_of_products_f32(
    mops_tensor_2d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_b,
    mops_tensor_1d_i32_t indexes
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::sparse_accumulation_of_products<float>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {tensor_a.data, {checked_cast(tensor_a.shape[0]), checked_cast(tensor_a.shape[1])}},
            {tensor_b.data, {checked_cast(tensor_b.shape[0]), checked_cast(tensor_b.shape[1])}},
            {indexes.data, {checked_cast(indexes.shape[0])}}
        );
    );
}


extern "C" int mops_cuda_sparse_accumulation_of_products_f64(
    mops_tensor_2d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_b,
    mops_tensor_1d_i32_t indexes
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::sparse_accumulation_of_products<double>(
            {output.data, {checked_cast(output.shape[0]), checked_cast(output.shape[1])}},
            {tensor_a.data, {checked_cast(tensor_a.shape[0]), checked_cast(tensor_a.shape[1])}},
            {tensor_b.data, {checked_cast(tensor_b.shape[0]), checked_cast(tensor_b.shape[1])}},
            {indexes.data, {checked_cast(indexes.shape[0])}}
        );
    );
}
