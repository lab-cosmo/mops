#include <stdexcept>

#include "mops/hpe.hpp"

template<typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    [[maybe_unused]] Tensor<scalar_t, 1> output,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_a,
    [[maybe_unused]] Tensor<scalar_t, 1> tensor_c,
    [[maybe_unused]] Tensor<int32_t, 2> p
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
