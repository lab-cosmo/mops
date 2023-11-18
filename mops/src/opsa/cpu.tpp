#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsa.hpp"
#include "mops/checks.hpp"

template<typename scalar_t>
void mops::outer_product_scatter_add(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_b,
    Tensor<int32_t, 1> indexes
) {
    // check_sizes(tensor_a, "A", 0, tensor_b, "B", 0, "opsa");
    // check_sizes(tensor_a, "A", 1, output, "O", 1, "opsa");
    // check_sizes(tensor_b, "B", 1, output, "O", 2, "opsa");
    // check_sizes(tensor_a, "A", 0, indexes, "P", 0, "opsa");
    // check_index_tensor(indexes, "P", output.shape[0], "opsa");

    if (!std::is_sorted(indexes.data, indexes.data + indexes.shape[0])) {
        throw std::runtime_error("`indexes` values should be sorted");
    }

    std::fill(output.data, output.data+output.shape[0]*output.shape[1], static_cast<scalar_t>(0.0));

    for (size_t i=0; i<tensor_a.shape[0]; i++) {
        auto i_output = indexes.data[i];
        for (size_t a_j=0; a_j<tensor_a.shape[1]; a_j++) {
            for (size_t b_j=0; b_j<tensor_b.shape[1]; b_j++) {
                auto output_index = tensor_b.shape[1] * (tensor_a.shape[1] * i_output + a_j) + b_j;
                output.data[output_index] += tensor_a.data[tensor_a.shape[1] * i + a_j]
                                           * tensor_b.data[tensor_b.shape[1] * i + b_j];
            }
        }
    }
}
