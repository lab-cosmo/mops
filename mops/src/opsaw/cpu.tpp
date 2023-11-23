#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsaw.hpp"

template<typename scalar_t>
void mops::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
) {
    // TODO
}
