#include <stdexcept>

#include "mops/opsaw.hpp"

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 3>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
