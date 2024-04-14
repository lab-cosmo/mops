#include <stdexcept>

#include "mops/sasaw.hpp"

template<typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 1>,
    Tensor<scalar_t, 3>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}

template<typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2> ,
    Tensor<scalar_t, 3>,
    Tensor<scalar_t, 3>,
    Tensor<scalar_t, 2> ,
    Tensor<scalar_t, 2> ,
    Tensor<scalar_t, 1> ,
    Tensor<scalar_t, 3> ,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}

template<typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 3> grad_W_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 3> grad_grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
