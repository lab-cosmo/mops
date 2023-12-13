#ifndef MOPS_HPE_HPP
#define MOPS_HPE_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/// TODO
template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A);

// these templates will be precompiled and provided in the mops library
extern template void
homogeneous_polynomial_evaluation(Tensor<float, 1> output, Tensor<float, 2> A,
                                  Tensor<float, 1> C,
                                  Tensor<int32_t, 2> indices_A);

extern template void
homogeneous_polynomial_evaluation(Tensor<double, 1> output, Tensor<double, 2> A,
                                  Tensor<double, 1> C,
                                  Tensor<int32_t, 2> indices_A);

/// TODO
template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation_vjp(
    Tensor<scalar_t, 2> grad_A, Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C, Tensor<int32_t, 2> indices_A);

// these templates will be precompiled and provided in the mops library
extern template void homogeneous_polynomial_evaluation_vjp(
    Tensor<float, 2> grad_A, Tensor<float, 1> grad_output, Tensor<float, 2> A,
    Tensor<float, 1> C, Tensor<int32_t, 2> indices_A);

extern template void homogeneous_polynomial_evaluation_vjp(
    Tensor<double, 2> grad_A, Tensor<double, 1> grad_output,
    Tensor<double, 2> A, Tensor<double, 1> C, Tensor<int32_t, 2> indices_A);

namespace cuda {
/// CUDA version of mops::homogeneous_polynomial_evaluation
template <typename scalar_t>
void MOPS_EXPORT homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A);
} // namespace cuda
} // namespace mops

#endif
