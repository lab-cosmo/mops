#include "mops/capi.hpp"

#include "mops/opsax.h"
#include "mops/opsax.hpp"

static size_t checked_cast(int64_t value) {
    if (value < 0 ||
        static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error("integer value '" + std::to_string(value) +
                                 "' does not fit in size_t");
    }
    return static_cast<size_t>(value);
}

extern "C" int mops_outer_product_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_r, mops_tensor_2d_f32_t tensor_x,
    mops_tensor_1d_i32_t tensor_i, mops_tensor_1d_i32_t tensor_j) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add_with_weights<float>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_r.data,
             {checked_cast(tensor_r.shape[0]),
              checked_cast(tensor_r.shape[1])}},
            {tensor_x.data,
             {checked_cast(tensor_x.shape[0]),
              checked_cast(tensor_x.shape[1])}},
            {tensor_i.data, {checked_cast(tensor_i.shape[0])}},
            {tensor_j.data, {checked_cast(tensor_j.shape[0])}}););
}

extern "C" int mops_outer_product_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_r, mops_tensor_2d_f64_t tensor_x,
    mops_tensor_1d_i32_t tensor_i, mops_tensor_1d_i32_t tensor_j) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add_with_weights<double>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_r.data,
             {checked_cast(tensor_r.shape[0]),
              checked_cast(tensor_r.shape[1])}},
            {tensor_x.data,
             {checked_cast(tensor_x.shape[0]),
              checked_cast(tensor_x.shape[1])}},
            {tensor_i.data, {checked_cast(tensor_i.shape[0])}},
            {tensor_j.data, {checked_cast(tensor_j.shape[0])}}););
}

extern "C" int mops_cuda_outer_product_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_r, mops_tensor_2d_f32_t tensor_x,
    mops_tensor_1d_i32_t tensor_i, mops_tensor_1d_i32_t tensor_j) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add_with_weights<float>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_r.data,
             {checked_cast(tensor_r.shape[0]),
              checked_cast(tensor_r.shape[1])}},
            {tensor_x.data,
             {checked_cast(tensor_x.shape[0]),
              checked_cast(tensor_x.shape[1])}},
            {tensor_i.data, {checked_cast(tensor_i.shape[0])}},
            {tensor_j.data, {checked_cast(tensor_j.shape[0])}}););
}

extern "C" int mops_cuda_outer_product_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_r, mops_tensor_2d_f64_t tensor_x,
    mops_tensor_1d_i32_t tensor_i, mops_tensor_1d_i32_t tensor_j) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add_with_weights<double>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {tensor_a.data,
             {checked_cast(tensor_a.shape[0]),
              checked_cast(tensor_a.shape[1])}},
            {tensor_r.data,
             {checked_cast(tensor_r.shape[0]),
              checked_cast(tensor_r.shape[1])}},
            {tensor_x.data,
             {checked_cast(tensor_x.shape[0]),
              checked_cast(tensor_x.shape[1])}},
            {tensor_i.data, {checked_cast(tensor_i.shape[0])}},
            {tensor_j.data, {checked_cast(tensor_j.shape[0])}}););
}
