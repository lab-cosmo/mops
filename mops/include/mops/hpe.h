#ifndef MOPS_OUTER_PRODUCT_SCATTER_ADD_H
#define MOPS_OUTER_PRODUCT_SCATTER_ADD_H

#include "mops/exports.h"
#include "mops/tensor.h"


#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::homogeneous_polynomial_evaluation for 32-bit floats
int MOPS_EXPORT mops_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_1d_f32_t tensor_c,
    mops_tensor_2d_i32_t p
);


/// CPU version of mops::homogeneous_polynomial_evaluation for 64-bit floats
int MOPS_EXPORT mops_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_1d_f64_t tensor_c,
    mops_tensor_2d_i32_t p
);


/// CUDA version of mops::homogeneous_polynomial_evaluation for 32-bit floats
int MOPS_EXPORT mops_cuda_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output,
    mops_tensor_2d_f32_t tensor_a,
    mops_tensor_1d_f32_t tensor_c,
    mops_tensor_2d_i32_t p
);


/// CUDA version of mops::homogeneous_polynomial_evaluation for 64-bit floats
int MOPS_EXPORT mops_cuda_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output,
    mops_tensor_2d_f64_t tensor_a,
    mops_tensor_1d_f64_t tensor_c,
    mops_tensor_2d_i32_t p
);


#ifdef __cplusplus
}
#endif


#endif
