#ifndef MOPS_OPSAW_HPP
#define MOPS_OPSAW_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/// TODO
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output, Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r, Tensor<scalar_t, 2> tensor_x,
    Tensor<int32_t, 1> i, Tensor<int32_t, 1> j);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add_with_weights(
    Tensor<float, 3> output, Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_r, Tensor<float, 2> tensor_x, Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j);

extern template void outer_product_scatter_add_with_weights(
    Tensor<double, 3> output, Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_r, Tensor<double, 2> tensor_x,
    Tensor<int32_t, 1> i, Tensor<int32_t, 1> j);

namespace cuda {
/// CUDA version of mops::outer_product_scatter_add_with
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output, Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r, Tensor<scalar_t, 2> tensor_x,
    Tensor<int32_t, 1> i, Tensor<int32_t, 1> j);
} // namespace cuda
} // namespace mops

#endif
