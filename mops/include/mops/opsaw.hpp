#ifndef MOPS_OPSAW_HPP
#define MOPS_OPSAW_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/// TODO
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

/// TODO
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

namespace cuda {
/// CUDA version of mops::outer_product_scatter_add_with_weights
template <typename scalar_t>
void MOPS_EXPORT outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
);

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
    Tensor<int32_t, 1> indices_output
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

} // namespace cuda
} // namespace mops

#endif
