#include <stdexcept>

#include "mops/opsax.hpp"

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r,
    Tensor<scalar_t, 2> tensor_x,
    Tensor<int32_t, 1> tensor_i,
    Tensor<int32_t, 1> tensor_j
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
