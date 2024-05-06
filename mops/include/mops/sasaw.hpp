#ifndef MOPS_SASAW_HPP
#define MOPS_SASAW_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/**
 * @brief Computes a fused sparse accumulation of products with weights and scatter addition.
 *
 * Computes the sparse accumulation of a 2D tensor A with a 1D tensor B, and scatters the result
 * into a 3D tensor W according to the indices in indices_W_1 and indices_W_2. The result is
 * then scattered into a 3D tensor output according to the indices in indices_output_1 and
 * indices_output_2.
 *
 * The operation can be described by the following pseudocode:
 * for j in range(J):
 *      for n in range(N):
 *          O[PO1[e], PO2[n], :] += A[e, PA[n]] * B[e, :] * C[n] * W[PW1[e], PW2[n], :]
 *
 * @param[out] output The output tensor of shape [output_size_1, output_size_2, size_A] where the
 * result will be stored.
 * @param[in] A Input tensor of shape [n_batch, size_A].
 * @param[in] B Input tensor of shape [n_batch, size_B].
 * @param[in] C Input tensor of shape [n_products].
 * @param[in] W Input tensor of shape [output_size_1, size_W, size_B].
 * @param[in] indices_A The indices of the first input tensor A where the elements will be
 *     taken for the product. The shape of this tensor is [n_products].
 * @param[in] indices_W_1 The indices of the first dimension of the output tensor W where the
 *     weights will be taken along the first dimension. The shape of this tensor is [n_scatter].
 * @param[in] indices_W_2 The indices of the second dimension of the output tensor W where the
 *     weights will be taken along the second dimension. The shape of this tensor is [n_products].
 * @param[in] indices_output_1 The indices of the first dimension of the output tensor where the
 *     result will be scattered. The shape of this tensor is [n_scatter].
 * @param[in] indices_output_2 The indices of the second dimension of the output tensor where the
 *     products will be accumulated. The shape of this tensor is [n_products].
 */
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_scatter_add_with_weights(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

/**
 * @brief Computes the vector-Jacobian product (vjp) of sparse_accumulation_scatter_add_with_weights
 * with respect to A, B, and W.
 *
 * @param[out] grad_A The gradient of the scalar objective with respect to A, of shape [n_batch, size_A].
 * @param[out] grad_B The gradient of the scalar objective with respect to B, of shape [n_batch, size_B].
 * @param[out] grad_W The gradient of the scalar objective with respect to W, of shape [output_size_1,
 * size_W, size_B].
 * @param[in] grad_output The gradient of the scalar objective with respect to the output, of shape
 *      [output_size_1, output_size_2, size_A].
 * @param[in] A See sparse_accumulation_scatter_add_with_weights.
 * @param[in] B See sparse_accumulation_scatter_add_with_weights.
 * @param[in] C See sparse_accumulation_scatter_add_with_weights.
 * @param[in] W See sparse_accumulation_scatter_add_with_weights.
 * @param[in] indices_A See sparse_accumulation_scatter_add_with_weights.
 * @param[in] indices_W_1 See sparse_accumulation_scatter_add_with_weights.
 * @param[in] indices_W_2 See sparse_accumulation_scatter_add_with_weights.
 * @param[in] indices_output_1 See sparse_accumulation_scatter_add_with_weights.
 * @param[in] indices_output_2 See sparse_accumulation_scatter_add_with_weights.
 */
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

/**
 * @brief Computes the vector-Jacobian product (vjp) of sparse_accumulation_scatter_add_with_weights
 * with respect to A, B, and W.
 *
 * @param[out] grad_grad_output The gradient of the scalar objective with respect to grad_output, of
 *     shape [output_size_1, output_size_2, size_A].
 * @param[out] grad_A_2 The gradient of the scalar objective with respect to A, of shape [n_batch,
 * size_A]. Not to be confused with grad_A in sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[out] grad_B_2 The gradient of the scalar objective with respect to B, of shape [n_batch,
 * size_B]. Not to be confused with grad_B in sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[out] grad_W_2 The gradient of the scalar objective with respect to W, of shape
 * [output_size_1, size_W, size_B]. Not to be confused with grad_W in
 * sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] grad_grad_A The gradient of the scalar objective with respect to grad_A, of shape
 * [n_batch, size_A].
 * @param[in] grad_grad_B The gradient of the scalar objective with respect to grad_B, of shape
 * [n_batch, size_B].
 * @param[in] grad_grad_W The gradient of the scalar objective with respect to grad_W, of shape
 * [output_size_1, size_W, size_B].
 * @param[in] grad_output See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] A See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] B See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] C See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] W See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] indices_A See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] indices_W_1 See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] indices_W_2 See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] indices_output_1 See sparse_accumulation_scatter_add_with_weights_vjp.
 * @param[in] indices_output_2 See sparse_accumulation_scatter_add_with_weights_vjp.
 */
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 3> grad_W_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 3> grad_grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 3> grad_W_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 3> grad_grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 3> grad_W_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 3> grad_grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

namespace cuda {
/// CUDA version of mops::sparse_accumulation_scatter_add_with
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

/// CUDA version of mops::sparse_accumulation_scatter_add_with_weights_vjp
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

/// TODO
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 3> grad_W_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 3> grad_grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 3> grad_W_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 3> grad_grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

extern template void sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 3> grad_W_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 3> grad_grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

} // namespace cuda
} // namespace mops

#endif
