#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsaw.hpp"
#include "mops/checks.hpp"
#include "mops/utils.hpp"


template<typename scalar_t>
void mops::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
) {

    check_sizes(A, "A", 0, B, "B", 0, "opsaw");
    check_sizes(A, "A", 1, output, "O", 1, "opsaw");
    check_sizes(B, "B", 1, output, "O", 2, "opsaw");
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, "opsaw");
    check_sizes(A, "A", 0, indices_W, "indices_W", 0, "opsaw");
    check_sizes(W, "W", 0, output, "O", 0, "opsaw");
    check_sizes(B, "B", 1, W, "W", 1, "opsaw");
    check_index_tensor(indices_output, "indices_output", output.shape[0], "opsaw");
    check_index_tensor(indices_W, "indices_W", output.shape[0], "opsaw");

    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = A.data;
    scalar_t* r_ptr = B.data;
    scalar_t* x_ptr = W.data;
    int* i_ptr = indices_output.data;
    int* j_ptr = indices_W.data;

    size_t E = indices_W.shape[0];
    size_t size_a = A.shape[1];
    size_t size_r = B.shape[1];

    std::vector<int32_t> first_occurrences = find_first_occurrences(i_ptr, E, output.shape[0]);
    std::fill(o_ptr, o_ptr+output.shape[0]*output.shape[1]*output.shape[2], static_cast<scalar_t>(0.0));

    #pragma omp parallel for 
    for (size_t i_position = 0; i_position < output.shape[0]; i_position++) {
        scalar_t* o_ptr_shifted_first_dim = o_ptr + i_position * size_a * size_r;
        int32_t index_j_start = first_occurrences[i_position];
        int32_t index_j_end = first_occurrences[i_position+1];
        for (int32_t e = index_j_start; e < index_j_end; e++) {
            scalar_t* a_ptr_shifted_first_dim = a_ptr + e * size_a;
            scalar_t* r_ptr_shifted_first_dim = r_ptr + e * size_r;
            scalar_t* x_ptr_shifted_first_dim = x_ptr + j_ptr[e] * size_r;
            for (size_t a_idx = 0; a_idx < size_a; a_idx++) {
                scalar_t current_a = a_ptr_shifted_first_dim[a_idx];
                scalar_t* o_ptr_shifted_second_dim = o_ptr_shifted_first_dim + a_idx * size_r;
                // Swapping the two inner loops might reduce the number of multiplications
                for (size_t r_idx = 0; r_idx < size_r; r_idx++) {
                    o_ptr_shifted_second_dim[r_idx] += current_a * r_ptr_shifted_first_dim[r_idx] * x_ptr_shifted_first_dim[r_idx];
                }
            }
        }
    }
}
