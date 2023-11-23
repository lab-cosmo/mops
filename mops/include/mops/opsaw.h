#ifndef MOPS_OPSAW_H
#define MOPS_OPSAW_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::outer_product_scatter_add_with for 32-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_r,
    mops_tensor_2d_f32_t tensor_x,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j
);


/// CPU version of mops::outer_product_scatter_add_with for 64-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_r,
    mops_tensor_2d_f64_t tensor_x,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j
);


/// CUDA version of mops::outer_product_scatter_add_with for 32-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_r,
    mops_tensor_2d_f32_t tensor_x,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j
);


/// CUDA version of mops::outer_product_scatter_add_with for 64-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_r,
    mops_tensor_2d_f64_t tensor_x,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j
);

#ifdef __cplusplus
}
#endif

#endif
