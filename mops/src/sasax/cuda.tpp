#include <stdexcept>

#include "mops/sasax.hpp"

template<typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights(
    [[maybe_unused]] Tensor<scalar_t, 3> output,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_a,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_r,
    [[maybe_unused]] Tensor<scalar_t, 3> tensor_x,
    [[maybe_unused]] Tensor<scalar_t, 1> tensor_c,
    [[maybe_unused]] Tensor<int, 1> tensor_i,
    [[maybe_unused]] Tensor<int, 1> tensor_j,
    [[maybe_unused]] Tensor<int, 1> tensor_m_1,
    [[maybe_unused]] Tensor<int, 1> tensor_m_2,
    [[maybe_unused]] Tensor<int, 1> tensor_m_3
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
