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

    long E = tensor_i.shape[0];
    long size_a = tensor_a.shape[1];
    long size_r = tensor_r.shape[1];

    for (long e = 0; e < E; e++) {
        scalar_t* o_ptr_shifted_first_dim = o_ptr + i_ptr[e] * size_a * size_r;
        scalar_t* a_ptr_shifted_first_dim = a_ptr + e * size_a;
        scalar_t* r_ptr_shifted_first_dim = r_ptr + e * size_r;
        scalar_t* x_ptr_shifted_first_dim = x_ptr + j_ptr[e] * size_r;
        for (long a_idx = 0; a_idx < size_a; a_idx++) {
            scalar_t current_a = a_ptr_shifted_first_dim[a_idx];
            scalar_t* o_ptr_shifted_second_dim = o_ptr_shifted_first_dim + a_idx * size_r;
            // Swapping the two inner loops might reduce the number of multiplications
            for (long r_idx = 0; r_idx < size_r; r_idx++) {
                o_ptr_shifted_second_dim[r_idx] += current_a * r_ptr_shifted_first_dim[r_idx] * x_ptr_shifted_first_dim[r_idx];
            }
        }
    }
}
