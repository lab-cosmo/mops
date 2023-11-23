#include <stdexcept>

#include "mops/opsaw.hpp"

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
