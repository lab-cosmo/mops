#ifndef MOPS_SAP_H
#define MOPS_SAP_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::sparse_accumulation_of_products for 32-bit floats
int MOPS_EXPORT mops_sparse_accumulation_of_products_f32(
    mops_tensor_2d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_b, mops_tensor_1d_f32_t tensor_c,
    mops_tensor_1d_i32_t p_a, mops_tensor_1d_i32_t p_b,
    mops_tensor_1d_i32_t p_o);

/// CPU version of mops::sparse_accumulation_of_products for 64-bit floats
int MOPS_EXPORT mops_sparse_accumulation_of_products_f64(
    mops_tensor_2d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_b, mops_tensor_1d_f64_t tensor_c,
    mops_tensor_1d_i32_t p_a, mops_tensor_1d_i32_t p_b,
    mops_tensor_1d_i32_t p_o);

/// CUDA version of mops::sparse_accumulation_of_products for 32-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_of_products_f32(
    mops_tensor_2d_f32_t output, mops_tensor_2d_f32_t tensor_a,
    mops_tensor_2d_f32_t tensor_b, mops_tensor_1d_f32_t tensor_c,
    mops_tensor_1d_i32_t p_a, mops_tensor_1d_i32_t p_b,
    mops_tensor_1d_i32_t p_o);

/// CUDA version of mops::sparse_accumulation_of_products for 64-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_of_products_f64(
    mops_tensor_2d_f64_t output, mops_tensor_2d_f64_t tensor_a,
    mops_tensor_2d_f64_t tensor_b, mops_tensor_1d_f64_t tensor_c,
    mops_tensor_1d_i32_t p_a, mops_tensor_1d_i32_t p_b,
    mops_tensor_1d_i32_t p_o);

#ifdef __cplusplus
}
#endif

#endif
