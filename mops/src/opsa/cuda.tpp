#include <stdexcept>

#include "mops/opsa.hpp"

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add(
    [[maybe_unused]] Tensor<scalar_t, 3> output,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_a,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_b,
    [[maybe_unused]] Tensor<int32_t, 1> indexes
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
