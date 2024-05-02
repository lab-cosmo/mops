#ifndef MOPS_SAP_HPP
#define MOPS_SAP_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/**
 * @brief Computes a sparse accumulation of products.
 *
 * Computes a sparse product of 2D tensors A, B along the last dimension of each, along
 * with coefficients. The products are accumulated in the second dimension of the output.
 *
 * The operation can be described by the following pseudocode:
 * for k in range(K):
 *      O[:, P_O[k]] += C[k] * A[:, P_A[k]] * B[:, P_B[k]]
 *
 * @param[out] output The output tensor of shape [n_batch, size_O] where the result
 *     will be stored.
 * @param[in] A Input tensor of shape [n_batch, size_A].
 * @param[in] B Input tensor of shape [n_batch, size_B].
 * @param[in] C Input tensor of shape [n_products].
 * @param[in] indices_A The indices of the first input tensor A where the elements will be
 *     taken for the product. The shape of this tensor is [n_products].
 * @param[in] indices_B The indices of the second input tensor B where the elements will be
 *     taken for the product. The shape of this tensor is [n_products].
 * @param[in] indices_output The indices of the output tensor where the products will be
 *     accumulated. The shape of this tensor is [n_products].
 */
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_of_products(
    Tensor<float, 2> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

extern template void sparse_accumulation_of_products(
    Tensor<double, 2> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

/**
 * @brief Computes the vector-Jacobian product (vjp) of sparse_accumulation_of_products with respect to A and B.
 *
 * @param[out] grad_A The gradient of the scalar objective with respect to A, of shape [n_batch, size_A].
 * @param[out] grad_B The gradient of the scalar objective with respect to B, of shape [n_batch, size_B].
 * @param[in] grad_output The gradient of the scalar objective with respect to the output, of shape
 *      [n_batch, size_O].
 * @param[in] A See sparse_accumulation_of_products.
 * @param[in] B See sparse_accumulation_of_products.
 * @param[in] C See sparse_accumulation_of_products.
 * @param[in] indices_A See sparse_accumulation_of_products.
 * @param[in] indices_B See sparse_accumulation_of_products.
 * @param[in] indices_output See sparse_accumulation_of_products.
 */
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_of_products_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_of_products_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

extern template void sparse_accumulation_of_products_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

/**
 * @brief Computes the vjp of the vjp of sparse_accumulation_of_products with respect to A and B.
 *
 * @param[out] grad_grad_output The gradient of the scalar objective with respect to grad_output, of
 *      shape [n_batch, size_O].
 * @param[out] grad_A_2 The gradient of the scalar objective with respect to A, of shape [n_batch,
 * size_A]. Not to be confused with grad_A in sparse_accumulation_of_products_vjp.
 * @param[out] grad_B_2 The gradient of the scalar objective with respect to B, of shape [n_batch,
 * size_B]. Not to be confused with grad_B in sparse_accumulation_of_products_vjp.
 * @param[in] grad_grad_A The gradient of the scalar objective with respect to grad_A, of shape
 *      [n_batch, size_A].
 * @param[in] grad_grad_B The gradient of the scalar objective with respect to grad_B, of shape
 *      [n_batch, size_B].
 * @param[in] grad_output See sparse_accumulation_of_products_vjp.
 * @param[in] A See sparse_accumulation_of_products_vjp.
 * @param[in] B See sparse_accumulation_of_products_vjp.
 * @param[in] C See sparse_accumulation_of_products_vjp.
 * @param[in] indices_A See sparse_accumulation_of_products_vjp.
 * @param[in] indices_B See sparse_accumulation_of_products_vjp.
 * @param[in] indices_output See sparse_accumulation_of_products_vjp.
 */
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_of_products_vjp_vjp(
    Tensor<scalar_t, 2> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_of_products_vjp_vjp(
    Tensor<float, 2> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

extern template void sparse_accumulation_of_products_vjp_vjp(
    Tensor<double, 2> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

namespace cuda {
/// CUDA version of mops::sparse_accumulation_of_products
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream = nullptr
);

extern template void sparse_accumulation_of_products(
    Tensor<float, 2> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

extern template void sparse_accumulation_of_products(
    Tensor<double, 2> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

/// CUDA version of mops::sparse_accumulation_of_products_vjp
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_of_products_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream = nullptr
);

extern template void sparse_accumulation_of_products_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

extern template void sparse_accumulation_of_products_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

/// TODO
template <typename scalar_t>
void MOPS_EXPORT sparse_accumulation_of_products_vjp_vjp(
    Tensor<scalar_t, 2> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream = nullptr
);

// these templates will be precompiled and provided in the mops library
extern template void sparse_accumulation_of_products_vjp_vjp(
    Tensor<float, 2> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

extern template void sparse_accumulation_of_products_vjp_vjp(
    Tensor<double, 2> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

} // namespace cuda
} // namespace mops

#endif
