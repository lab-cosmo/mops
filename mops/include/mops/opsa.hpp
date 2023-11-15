#ifndef MOPS_OPSA_HPP
#define MOPS_OPSA_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"
#include "mops/utils.hpp"

namespace mops {
/// TODO
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add(Tensor<scalar_t, 2> output,
                                           Tensor<scalar_t, 2> tensor_a,
                                           Tensor<scalar_t, 2> tensor_b,
                                           Tensor<int32_t, 1> indexes);

// these templates will be precompiled and provided in the mops library
extern template void outer_product_scatter_add(Tensor<float, 2> output,
                                               Tensor<float, 2> tensor_a,
                                               Tensor<float, 2> tensor_b,
                                               Tensor<int32_t, 1> indexes);

extern template void outer_product_scatter_add(Tensor<double, 2> output,
                                               Tensor<double, 2> tensor_a,
                                               Tensor<double, 2> tensor_b,
                                               Tensor<int32_t, 1> indexes);

namespace cuda {
/// CUDA version of mops::outer_product_scatter_add
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add(Tensor<scalar_t, 2> output,
                                           Tensor<scalar_t, 2> tensor_a,
                                           Tensor<scalar_t, 2> tensor_b,
                                           Tensor<int32_t, 1> indexes);
} // namespace cuda
} // namespace mops

#endif
