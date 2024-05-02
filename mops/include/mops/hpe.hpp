#ifndef MOPS_HPE_HPP
#define MOPS_HPE_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/**
 * @brief Computes a homogeneous polynomial on a set of n_batch points.
 *
 * The operation can be described by the following pseudocode:
 * for j in range(J):
 *      O[:] += C[j] * A[:, P_1[j, 1]] * A[:, P_2[j, 2]] * ...
 *
 * @param[out] output The output tensor of shape [n_batch] where the result will be stored.
 * @param[in] A The input tensor of shape [n_batch, n_factors] containing the individual
 *      input factors that will be multiplied into the monomials.
 * @param[in] C The coefficients of the monomials, of shape [n_monomials].
 * @param[in] indices_A The indices of the factors in A that will be multiplied together
 *     for each monomial. The shape of this tensor is [n_monomials, polynomial_degree].
 */
template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C, Tensor<int32_t, 2> indices_A
);

// these templates will be precompiled and provided in the mops library
extern template void homogeneous_polynomial_evaluation(
    Tensor<float, 1> output, Tensor<float, 2> A, Tensor<float, 1> C, Tensor<int32_t, 2> indices_A
);

extern template void homogeneous_polynomial_evaluation(
    Tensor<double, 1> output, Tensor<double, 2> A, Tensor<double, 1> C, Tensor<int32_t, 2> indices_A
);

/**
 * @brief Computes the vector-Jacobian product (vjp) of homogeneous_polynomial_evaluation with
 * respect to A.
 *
 * @param[out] grad_A The gradient of the scalar objective with respect to A, of shape [n_batch,
 * n_factors].
 * @param[in] grad_output The gradient of the scalar objective with respect to the output, of shape
 *      [n_batch].
 * @param[in] A See homogeneous_polynomial_evaluation.
 * @param[in] C See homogeneous_polynomial_evaluation.
 * @param[in] indices_A See homogeneous_polynomial_evaluation.
 */
template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A
);

// these templates will be precompiled and provided in the mops library
extern template void homogeneous_polynomial_evaluation_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A
);

extern template void homogeneous_polynomial_evaluation_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);

/**
 * @brief Computes the vjp of the vjp of homogeneous_polynomial_evaluation with respect to A.
 *
 * @param[out] grad_grad_output The gradient of the scalar objective with respect to grad_output, of
 *      shape [n_batch].
 * @param[out] grad_A_2 The gradient of the scalar objective with respect to A, of shape [n_batch,
 * n_factors]. Not to be confused with grad_A in homogeneous_polynomial_evaluation_vjp.
 * @param[in] grad_grad_A The gradient of the scalar objective with respect to grad_A, of shape
 *      [n_batch, n_factors].
 * @param[in] grad_output See homogeneous_polynomial_evaluation_vjp.
 * @param[in] A See homogeneous_polynomial_evaluation_vjp.
 * @param[in] C See homogeneous_polynomial_evaluation_vjp.
 * @param[in] indices_A See homogeneous_polynomial_evaluation_vjp.
 */
template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation_vjp_vjp(
    Tensor<scalar_t, 1> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A
);

extern template void homogeneous_polynomial_evaluation_vjp_vjp(
    Tensor<float, 1> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A
);

extern template void homogeneous_polynomial_evaluation_vjp_vjp(
    Tensor<double, 1> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);

namespace cuda {
/// CUDA version of mops::homogeneous_polynomial_evaluation
template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C, Tensor<int32_t, 2> indices_A, void* cuda_stream = nullptr
);

extern template void homogeneous_polynomial_evaluation(
    Tensor<float, 1> output, Tensor<float, 2> A, Tensor<float, 1> C, Tensor<int32_t, 2> indices_A, void* cuda_stream
);

extern template void homogeneous_polynomial_evaluation(
    Tensor<double, 1> output, Tensor<double, 2> A, Tensor<double, 1> C, Tensor<int32_t, 2> indices_A, void* cuda_stream
);

template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A,
    void* cuda_stream = nullptr
);

extern template void homogeneous_polynomial_evaluation_vjp(
    Tensor<float, 2> grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A,
    void* cuda_stream
);

extern template void homogeneous_polynomial_evaluation_vjp(
    Tensor<double, 2> grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A,
    void* cuda_stream
);

template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation_vjp_vjp(
    Tensor<scalar_t, 1> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A,
    void* cuda_stream = nullptr
);

extern template void homogeneous_polynomial_evaluation_vjp_vjp(
    Tensor<float, 1> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A,
    void* cuda_stream
);

extern template void homogeneous_polynomial_evaluation_vjp_vjp(
    Tensor<double, 1> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A,
    void* cuda_stream
);

} // namespace cuda
} // namespace mops

#endif
