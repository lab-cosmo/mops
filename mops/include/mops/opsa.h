#ifndef MOPS_OPSA_H
#define MOPS_OPSA_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::outer_product_scatter_add for 32-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_f32(
    mops_tensor_2d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_b, mops_tensor_1d_i32_t indexes);

/// CPU version of mops::outer_product_scatter_add for 64-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_f64(
    mops_tensor_2d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_b, mops_tensor_1d_i32_t indexes);

/// CUDA version of mops::outer_product_scatter_add for 32-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_f32(
    mops_tensor_2d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_b, mops_tensor_1d_i32_t indexes);

/// CUDA version of mops::outer_product_scatter_add for 64-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_f64(
    mops_tensor_2d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_b, mops_tensor_1d_i32_t indexes);

#ifdef __cplusplus
}
#endif

#endif
