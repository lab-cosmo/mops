#include <stdexcept>

#include "mops/hpe.hpp"

template<typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
