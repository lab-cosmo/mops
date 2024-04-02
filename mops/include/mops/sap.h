#ifndef MOPS_SAP_H
#define MOPS_SAP_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::sparse_accumulation_of_products for 32-bit floats
int MOPS_EXPORT mops_sparse_accumulation_of_products_f32(
    mops_tensor_2d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_f32_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::sparse_accumulation_of_products for 64-bit floats
int MOPS_EXPORT mops_sparse_accumulation_of_products_f64(
    mops_tensor_2d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_f64_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::sparse_accumulation_of_products for 32-bit floats
int MOPS_EXPORT mops_sparse_accumulation_of_products_vjp_f32(
    mops_tensor_2d_f32_t grad_A,
    mops_tensor_2d_f32_t grad_B,
    mops_tensor_2d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_f32_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::sparse_accumulation_of_products for 64-bit floats
int MOPS_EXPORT mops_sparse_accumulation_of_products_vjp_f64(
    mops_tensor_2d_f64_t grad_A,
    mops_tensor_2d_f64_t grad_B,
    mops_tensor_2d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_f64_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::sparse_accumulation_of_products for 32-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_of_products_f32(
    mops_tensor_2d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_f32_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::sparse_accumulation_of_products for 64-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_of_products_f64(
    mops_tensor_2d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_f64_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::sparse_accumulation_of_products_vjp for 32-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_of_products_vjp_f32(
    mops_tensor_2d_f32_t grad_A,
    mops_tensor_2d_f32_t grad_B,
    mops_tensor_2d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_f32_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::sparse_accumulation_of_products_vjp for 64-bit floats
int MOPS_EXPORT mops_cuda_sparse_accumulation_of_products_vjp_f64(
    mops_tensor_2d_f64_t grad_A,
    mops_tensor_2d_f64_t grad_B,
    mops_tensor_2d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_f64_t C,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_B,
    mops_tensor_1d_i32_t indices_output
);

#ifdef __cplusplus
}
#endif

#endif
