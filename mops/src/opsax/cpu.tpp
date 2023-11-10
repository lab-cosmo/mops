#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsax.hpp"

template<typename scalar_t>
void mops::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r,
    Tensor<scalar_t, 2> tensor_x,
    Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j
) {
    // TODO
}
