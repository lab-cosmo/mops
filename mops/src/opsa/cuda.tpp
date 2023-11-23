#include <stdexcept>

#include "mops/opsa.hpp"

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
