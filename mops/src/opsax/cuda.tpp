#include <stdexcept>

#include "mops/opsax.hpp"

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add_with_weights(
    [[maybe_unused]] Tensor<scalar_t, 3> output,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_a,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_r,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_x,
    [[maybe_unused]] Tensor<int32_t, 1> tensor_i,
    [[maybe_unused]] Tensor<int32_t, 1> tensor_j
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
