#ifndef MOPS_OPSA_HPP
#define MOPS_OPSA_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/// TODO
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add(Tensor<scalar_t, 2> output,
                                           Tensor<scalar_t, 2> A,
                                           Tensor<scalar_t, 2> B,
                                           Tensor<int32_t, 1> indices_output);

// these templates will be precompiled and provided in the mops library
extern template void
outer_product_scatter_add(Tensor<float, 2> output, Tensor<float, 2> A,
                          Tensor<float, 2> B,
                          Tensor<int32_t, 1> indices_output);

extern template void
outer_product_scatter_add(Tensor<double, 2> output, Tensor<double, 2> A,
                          Tensor<double, 2> B,
                          Tensor<int32_t, 1> indices_output);

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
    Tensor<scalar_t, 2> grad_A, Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output, Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B, Tensor<int32_t, 1> indices_output);

// these templates will be precompiled and provided in the mops library
extern template void
outer_product_scatter_add_vjp(Tensor<float, 2> grad_A, Tensor<float, 2> grad_B,
                              Tensor<float, 2> grad_output, Tensor<float, 2> A,
                              Tensor<float, 2> B,
                              Tensor<int32_t, 1> indices_output);

extern template void outer_product_scatter_add_vjp(
    Tensor<double, 2> grad_A, Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output);

namespace cuda {
/// CUDA version of mops::outer_product_scatter_add
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add(Tensor<scalar_t, 2> output,
                                           Tensor<scalar_t, 2> A,
                                           Tensor<scalar_t, 2> B,
                                           Tensor<int32_t, 1> first_occurences,
                                           Tensor<int32_t, 1> indices_output);

extern template void outer_product_scatter_add(
    Tensor<float, 2> output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<int32_t, 1> first_occurences, Tensor<int32_t, 1> indices_output);

extern template void outer_product_scatter_add(
    Tensor<double, 2> output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<int32_t, 1> first_occurences, Tensor<int32_t, 1> indices_output);

} // namespace cuda
} // namespace mops

#endif
