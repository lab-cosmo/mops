#ifndef MOPS_OPSA_H
#define MOPS_OPSA_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::outer_product_scatter_add for 32-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_f32(
    mops_tensor_3d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::outer_product_scatter_add for 64-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_f64(
    mops_tensor_3d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::outer_product_scatter_add_vjp for 32-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_vjp_f32(
    mops_tensor_2d_f32_t grad_A,
    mops_tensor_2d_f32_t grad_B,
    mops_tensor_3d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::outer_product_scatter_add_vjp for 64-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_vjp_f64(
    mops_tensor_2d_f64_t grad_A,
    mops_tensor_2d_f64_t grad_B,
    mops_tensor_3d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::outer_product_scatter_add_vjp_vjp for 32-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_vjp_vjp_f32(
    mops_tensor_3d_f32_t grad_grad_output,
    mops_tensor_2d_f32_t grad_A_2,
    mops_tensor_2d_f32_t grad_B_2,
    mops_tensor_2d_f32_t grad_grad_A,
    mops_tensor_2d_f32_t grad_grad_B,
    mops_tensor_3d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CPU version of mops::outer_product_scatter_add_vjp_vjp for 64-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_vjp_vjp_f64(
    mops_tensor_3d_f64_t grad_grad_output,
    mops_tensor_2d_f64_t grad_A_2,
    mops_tensor_2d_f64_t grad_B_2,
    mops_tensor_2d_f64_t grad_grad_A,
    mops_tensor_2d_f64_t grad_grad_B,
    mops_tensor_3d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::outer_product_scatter_add for 32-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_f32(
    mops_tensor_3d_f32_t output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::outer_product_scatter_add for 64-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_f64(
    mops_tensor_3d_f64_t output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::outer_product_scatter_add_vjp for 32-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_vjp_f32(
    mops_tensor_2d_f32_t grad_A,
    mops_tensor_2d_f32_t grad_B,
    mops_tensor_3d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::outer_product_scatter_add_vjp for 64-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_vjp_f64(
    mops_tensor_2d_f64_t grad_A,
    mops_tensor_2d_f64_t grad_B,
    mops_tensor_3d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::outer_product_scatter_add_vjp_vjp for 32-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_vjp_vjp_f32(
    mops_tensor_3d_f32_t grad_grad_output,
    mops_tensor_2d_f32_t grad_A_2,
    mops_tensor_2d_f32_t grad_B_2,
    mops_tensor_2d_f32_t grad_grad_A,
    mops_tensor_2d_f32_t grad_grad_B,
    mops_tensor_3d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_i32_t indices_output
);

/// CUDA version of mops::outer_product_scatter_add_vjp_vjp for 64-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_vjp_vjp_f64(
    mops_tensor_3d_f64_t grad_grad_output,
    mops_tensor_2d_f64_t grad_A_2,
    mops_tensor_2d_f64_t grad_B_2,
    mops_tensor_2d_f64_t grad_grad_A,
    mops_tensor_2d_f64_t grad_grad_B,
    mops_tensor_3d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_i32_t indices_output
);

#ifdef __cplusplus
}
#endif

#endif
