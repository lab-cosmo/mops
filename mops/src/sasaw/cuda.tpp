#include <stdexcept>

#include "mops/sasaw.hpp"

template<typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 1>,
    Tensor<scalar_t, 3>,
    Tensor<int, 1>,
    Tensor<int, 1>,
    Tensor<int, 1>,
    Tensor<int, 1>,
    Tensor<int, 1>
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
