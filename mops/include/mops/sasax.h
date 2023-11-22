#ifndef MOPS_SASAX_H
#define MOPS_SASAX_H

#include "mops/exports.h"
#include "mops/tensor.h"


#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::sparse_accumulation_scatter_add_with for 32-bit floats
int MOPS_EXPORT mops_sparse_accumulation_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_r,
    mops_tensor_3d_f32_t tensor_x,
    mops_tensor_1d_f32_t tensor_c,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j,
    mops_tensor_1d_i32_t tensor_m_1,
    mops_tensor_1d_i32_t tensor_m_2,
    mops_tensor_1d_i32_t tensor_m_3
);


/// CPU version of mops::sparse_accumulation_scatter_add_with for 64-bit floats
int MOPS_EXPORT mops_sparse_accumulation_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_r,
    mops_tensor_3d_f64_t tensor_x,
    mops_tensor_1d_f64_t tensor_c,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j,
    mops_tensor_1d_i32_t tensor_m_1,
    mops_tensor_1d_i32_t tensor_m_2,
    mops_tensor_1d_i32_t tensor_m_3
);


/// CUDA version of mops::sparse_accumulation_scatter_add_with for 32-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_r,
    mops_tensor_3d_f32_t tensor_x,
    mops_tensor_1d_f32_t tensor_c,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j,
    mops_tensor_1d_i32_t tensor_m_1,
    mops_tensor_1d_i32_t tensor_m_2,
    mops_tensor_1d_i32_t tensor_m_3
);


/// CUDA version of mops::sparse_accumulation_scatter_add_with for 64-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_r,
    mops_tensor_3d_f64_t tensor_x,
    mops_tensor_1d_f64_t tensor_c,
    mops_tensor_1d_i32_t tensor_i,
    mops_tensor_1d_i32_t tensor_j,
    mops_tensor_1d_i32_t tensor_m_1,
    mops_tensor_1d_i32_t tensor_m_2,
    mops_tensor_1d_i32_t tensor_m_3
);


#ifdef __cplusplus
}
#endif


#endif
