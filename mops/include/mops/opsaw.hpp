#ifndef MOPS_OPSAW_HPP
#define MOPS_OPSAW_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/**
 * @brief Computes a fused outer product and scatter-add operation with weights.
 *
 * Computes the outer product between 2D tensors A, B along the last dimension of each,
 * and performs a scatter add operation in the first dimension according to the indices
 * in indices_output. The outer products are weighted by the values in W.
 *
 * The operation can be described by the following pseudocode:
 * for j in range(J):
 *      O[PO[j], :, :] += A[j, :, None] * B[j, None, :] * W[PW[j], None, :]
 *
 * @param[out] output The output tensor of shape [output_size, size_A, size_B] where the result
 *      will be stored.
 * @param[in] A Input tensor of shape [n_batch, size_A].
 * @param[in] B Input tensor of shape [n_batch, size_B].
 * @param[in] W Input tensor of shape [output_size, size_B] containing the weights for each outer product.
 * @param[in] indices_W The indices of the weights in W that will be used for each outer product.
 *     The shape of this tensor is [n_batch].
 * @param[in] indices_output The indices of the output tensor where the outer products will be
 *     added. The shape of this tensor is [n_batch].
 */
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_with_weights(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add_with_weights(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

/**
 * @brief Computes the vector-Jacobian product (vjp) of outer_product_scatter_add_with_weights with
 * respect to A, B and W.
 *
 * @param[out] grad_A The gradient of the scalar objective with respect to A, of shape [n_batch, size_A].
 * @param[out] grad_B The gradient of the scalar objective with respect to B, of shape [n_batch, size_B].
 * @param[out] grad_W The gradient of the scalar objective with respect to W, of shape [n_batch].
 * @param[in] grad_output The gradient of the scalar objective with respect to the output, of shape
 *      [output_size, size_A, size_B].
 * @param[in] A See outer_product_scatter_add_with_weights.
 * @param[in] B See outer_product_scatter_add_with_weights.
 * @param[in] W See outer_product_scatter_add_with_weights.
 * @param[in] indices_W See outer_product_scatter_add_with_weights.
 * @param[in] indices_output See outer_product_scatter_add_with_weights.
 */
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_with_weights_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add_with_weights_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

/**
 * @brief Computes the vjp of the vjp of outer_product_scatter_add_with_weights with respect to A,
 * B, and W.
 *
 * @param[out] grad_grad_output The gradient of the scalar objective with respect to grad_output, of
 *      shape [output_size, size_A, size_B].
 * @param[out] grad_A_2 The gradient of the scalar objective with respect to A, of shape [n_batch,
 * size_A]. Not to be confused with grad_A in outer_product_scatter_add_with_weights_vjp.
 * @param[out] grad_B_2 The gradient of the scalar objective with respect to B, of shape [n_batch,
 * size_B]. Not to be confused with grad_B in outer_product_scatter_add_with_weights_vjp.
 * @param[out] grad_W_2 The gradient of the scalar objective with respect to W, of shape [n_batch].
 *      Not to be confused with grad_W in outer_product_scatter_add_with_weights_vjp.
 * @param[in] grad_grad_A The gradient of the scalar objective with respect to grad_A, of shape
 *      [n_batch, size_A].
 * @param[in] grad_grad_B The gradient of the scalar objective with respect to grad_B, of shape
 *      [n_batch, size_B].
 * @param[in] grad_grad_W The gradient of the scalar objective with respect to grad_W, of shape
 *      [n_batch].
 * @param[in] grad_output See outer_product_scatter_add_with_weights_vjp.
 * @param[in] A See outer_product_scatter_add_with_weights_vjp.
 * @param[in] B See outer_product_scatter_add_with_weights_vjp.
 * @param[in] W See outer_product_scatter_add_with_weights_vjp.
 * @param[in] indices_W See outer_product_scatter_add_with_weights_vjp.
 * @param[in] indices_output See outer_product_scatter_add_with_weights_vjp.
 */
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_W_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 2> grad_grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_with_weights_vjp_vjp(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_W_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 2> grad_grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

extern template void outer_product_scatter_add_with_weights_vjp_vjp(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_W_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 2> grad_grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

namespace cuda {
/// CUDA version of mops::outer_product_scatter_add_with_weights
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream = nullptr
);

extern template void outer_product_scatter_add_with_weights(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

extern template void outer_product_scatter_add_with_weights(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

/// CUDA version of mops::outer_product_scatter_add_with_weights_vjp
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream = nullptr
);

extern template void outer_product_scatter_add_with_weights_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

extern template void outer_product_scatter_add_with_weights_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

/// TODO
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_W_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 2> grad_grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream = nullptr
);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_with_weights_vjp_vjp(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_W_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 2> grad_grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

extern template void outer_product_scatter_add_with_weights_vjp_vjp(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_W_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 2> grad_grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

} // namespace cuda
} // namespace mops

#endif
