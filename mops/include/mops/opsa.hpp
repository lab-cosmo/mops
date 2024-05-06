#ifndef MOPS_OPSA_HPP
#define MOPS_OPSA_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/**
 * @brief Computes a fused outer product and scatter-add operation.
 *
 * Computes the outer product between 2D tensors A, B along the last dimension of each,
 * and performs a scatter add operation in the first dimension according to the indices
 * in indices_output.
 *
 * For example, If A has shape (5, 32) and B has shape (5, 16), and indices_output contains
 * [0, 0, 1, 1, 2], the output will have shape (output_size, 32, 16), where output_size must
 * be greater than or equal to 3.
 *
 * The operation can be described by the following pseudocode:
 * for j in range(J):
 *      O[P[j], :, :] += A[j, :, None] * B[j, None, :]
 *
 * @param[out] output The output tensor of shape [output_size, size_A, size_B] where the result
 *      will be stored.
 * @param[in] A Input tensor of shape [n_batch, size_A].
 * @param[in] B Input tensor of shape [n_batch, size_B].
 * @param[in] indices_output The indices of the output tensor where the outer products will be
 *     added. The shape of this tensor is [n_batch].
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

/**
 * @brief Computes the vector-Jacobian product (vjp) of outer_product_scatter_add with respect to A and B.
 *
 * @param[out] grad_A The gradient of the scalar objective with respect to A, of shape [n_batch, size_A].
 * @param[out] grad_B The gradient of the scalar objective with respect to B, of shape [n_batch, size_B].
 * @param[in] grad_output The gradient of the scalar objective with respect to the output, of shape
 *      [output_size, size_A, size_B].
 * @param[in] A See outer_product_scatter_add.
 * @param[in] B See outer_product_scatter_add.
 * @param[in] indices_output See outer_product_scatter_add.
 */
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

/**
 * @brief Computes the vjp of the vjp of outer_product_scatter_add with respect to A and B.
 *
 * @param[out] grad_grad_output The gradient of the scalar objective with respect to grad_output, of
 *      shape [output_size, size_A, size_B].
 * @param[out] grad_A_2 The gradient of the scalar objective with respect to A, of shape [n_batch,
 * size_A]. Not to be confused with grad_A in outer_product_scatter_add_vjp.
 * @param[out] grad_B_2 The gradient of the scalar objective with respect to B, of shape [n_batch,
 * size_B]. Not to be confused with grad_B in outer_product_scatter_add_vjp.
 * @param[in] grad_grad_A The gradient of the scalar objective with respect to grad_A, of shape
 *      [n_batch, size_A].
 * @param[in] grad_grad_B The gradient of the scalar objective with respect to grad_B, of shape
 *      [n_batch, size_B].
 * @param[in] grad_output See outer_product_scatter_add_vjp.
 * @param[in] A See outer_product_scatter_add_vjp.
 * @param[in] B See outer_product_scatter_add_vjp.
 * @param[in] indices_output See outer_product_scatter_add_vjp.
 */
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
