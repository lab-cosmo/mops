#ifndef MOPS_SASAW_HPP
#define MOPS_SASAW_HPP

#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
/// TODO
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

/// TODO
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

} // namespace cuda
} // namespace mops

#endif
