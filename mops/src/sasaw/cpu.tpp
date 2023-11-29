#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sasaw.hpp"

template<typename scalar_t>
void mops::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1,
    Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1,
    Tensor<int, 1> indices_output_2
) {
    // TODO
}


template<typename scalar_t>
void mops::sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1,
    Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1,
    Tensor<int, 1> indices_output_2
) {
    // TODO
}
