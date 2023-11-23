#ifndef MOPS_HPE_H
#define MOPS_HPE_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::homogeneous_polynomial_evaluation for 32-bit floats
int MOPS_EXPORT mops_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_1d_f32_t C,
    mops_tensor_2d_i32_t indices_A
);


/// CPU version of mops::homogeneous_polynomial_evaluation for 64-bit floats
int MOPS_EXPORT mops_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_1d_f64_t C,
    mops_tensor_2d_i32_t indices_A
);


/// CUDA version of mops::homogeneous_polynomial_evaluation for 32-bit floats
int MOPS_EXPORT mops_cuda_homogeneous_polynomial_evaluation_f32(
    mops_tensor_1d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_1d_f32_t C,
    mops_tensor_2d_i32_t indices_A
);


/// CUDA version of mops::homogeneous_polynomial_evaluation for 64-bit floats
int MOPS_EXPORT mops_cuda_homogeneous_polynomial_evaluation_f64(
    mops_tensor_1d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_1d_f64_t C,
    mops_tensor_2d_i32_t indices_A
);

#ifdef __cplusplus
}
#endif

#endif
