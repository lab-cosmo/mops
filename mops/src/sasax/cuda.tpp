#include <stdexcept>

#include "mops/sasax.hpp"

template<typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r,
    Tensor<scalar_t, 3> tensor_x,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int, 1> tensor_i,
    Tensor<int, 1> tensor_j,
    Tensor<int, 1> tensor_m_1,
    Tensor<int, 1> tensor_m_2,
    Tensor<int, 1> tensor_m_3
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
