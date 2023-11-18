#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsax.hpp"
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
    // if (tensor_a.shape[0] != tensor_b.shape[0]) {
    //     throw std::runtime_error(
    //         "A and B tensors must have the same number of elements along the "
    //         "first dimension, got " + std::to_string(tensor_a.shape[0]) + " and " +
    //         std::to_string(tensor_b.shape[0])
    //     );
    // }

    // if (tensor_a.shape[0] != indexes.shape[0]) {
    //     throw std::runtime_error(
    //         "indexes must contain the same number of elements as the first "
    //         "dimension of A and B , got " + std::to_string(indexes.shape[0]) +
    //         " and " + std::to_string(tensor_a.shape[0])
    //     );
    // }

    // if (tensor_a.shape[1] * tensor_b.shape[1] != output.shape[1]) {
    //     throw std::runtime_error(
    //         "output tensor must have space for " + std::to_string(tensor_a.shape[1] * tensor_b.shape[1]) +
    //         " along the second dimension, got " + std::to_string(output.shape[1])
    //     );
    // }

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
