#ifndef MOPS_OPSA_HPP
#define MOPS_OPSA_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/*
 * Outer-Product-Scatter-Add (OPSA)
 * Computes the outer product between tensors A, B along the last dimension, and sums the result
 * into a new tensor of shape [output_size, A.shape[1], B.shape[1]], where the summation index
 * is given by the tensor indices_output.
 *
 * For example, If A has shape (5, 32) and B has shape (5, 16), and indices_output contains
 * [0, 0, 1, 1, 2], the output will have shape (3, 32, 16). For clarity, the
 * value of output[0] in this case would be equal to
 * output[0, :, :] = A[0, :, None] * B[0, None, :] + A[1, :, None] * B[1, None,  :]
 */
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B, Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add(
    Tensor<double, 3> output, Tensor<double, 2> A, Tensor<double, 2> B, Tensor<int32_t, 1> indices_output
);

/// Vector-Jacobian product for `outer_product_scatter_add` (i.e. backward
/// propagation of gradients)
///
/// `grad_A` and `grad_B` are the outputs of this function, and should have
/// the same shape as `A` and `B`. If you don't need one of these gradients,
/// set the corresponding `.data` pointer to `NULL`.
///
/// `grad_output` should have the same shape as `output` in
/// `outer_product_scatter_add`.
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);

/// TODO
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_vjp_vjp(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add_vjp_vjp(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);

namespace cuda {
/// CUDA version of mops::outer_product_scatter_add
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B, Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add(
    Tensor<double, 3> output, Tensor<double, 2> A, Tensor<double, 2> B, Tensor<int32_t, 1> indices_output
);

template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);

/// TODO
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_vjp_vjp(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add_vjp_vjp(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);

} // namespace cuda
} // namespace mops

#endif
