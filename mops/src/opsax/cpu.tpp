#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <execution>
#include <numeric>
#include <vector>

#include "mops/opsax.hpp"
#include "mops/checks.hpp"
#include "mops/utils.hpp"


template<typename scalar_t>
void mops::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r,
    Tensor<scalar_t, 2> tensor_x,
    Tensor<int32_t, 1> tensor_i,
    Tensor<int32_t, 1> tensor_j
) {
    check_sizes(tensor_a, "A", 0, tensor_r, "R", 0, "opsax");
    check_sizes(tensor_a, "A", 1, output, "O", 1, "opsax");
    check_sizes(tensor_r, "R", 1, output, "O", 2, "opsax");
    check_sizes(tensor_a, "A", 0, tensor_i, "I", 0, "opsax");
    check_sizes(tensor_a, "A", 0, tensor_j, "J", 0, "opsax");
    check_sizes(tensor_x, "X", 0, output, "O", 0, "opsax");
    check_sizes(tensor_r, "R", 1, tensor_x, "X", 1, "opsax");
    check_index_tensor(tensor_i, "I", output.shape[0], "opsax");
    check_index_tensor(tensor_j, "J", output.shape[0], "opsax");

    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = tensor_a.data;
    scalar_t* r_ptr = tensor_r.data;
    scalar_t* x_ptr = tensor_x.data;
    int* i_ptr = tensor_i.data;
    int* j_ptr = tensor_j.data;

    size_t E = tensor_i.shape[0];
    size_t size_a = tensor_a.shape[1];
    size_t size_r = tensor_r.shape[1];

    std::vector<int32_t> first_occurrences = find_first_occurrences(i_ptr, E, output.shape[0]);
    std::fill(o_ptr, o_ptr+output.shape[0]*output.shape[1]*output.shape[2], static_cast<scalar_t>(0.0));

    std::vector<size_t> indices(output.shape[0]);
    std::iota(indices.begin(), indices.end(), 0); // Fill with consecutive numbers
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i_position) {
        scalar_t* o_ptr_shifted_first_dim = o_ptr + i_position * size_a * size_r;
        int32_t index_j_start = first_occurrences[i_position];
        int32_t index_j_end = first_occurrences[i_position + 1];
        for (int32_t e = index_j_start; e < index_j_end; e++) {
            scalar_t* a_ptr_shifted_first_dim = a_ptr + e * size_a;
            scalar_t* r_ptr_shifted_first_dim = r_ptr + e * size_r;
            scalar_t* x_ptr_shifted_first_dim = x_ptr + j_ptr[e] * size_r;
            for (size_t a_idx = 0; a_idx < size_a; a_idx++) {
                scalar_t current_a = a_ptr_shifted_first_dim[a_idx];
                scalar_t* o_ptr_shifted_second_dim = o_ptr_shifted_first_dim + a_idx * size_r;
                for (size_t r_idx = 0; r_idx < size_r; r_idx++) {
                    o_ptr_shifted_second_dim[r_idx] += current_a * r_ptr_shifted_first_dim[r_idx] * x_ptr_shifted_first_dim[r_idx];
                }
            }
        }
    });
}
