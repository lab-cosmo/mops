#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sasax.hpp"
#include "mops/utils.hpp"

template<typename scalar_t>
void mops::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r,
    Tensor<scalar_t, 3> tensor_x,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int, 1> tensor_i,
    Tensor<int, 1> tensor_j,
    Tensor<int, 1> tensor_m_1,
    Tensor<int, 1> tensor_m_2,
    Tensor<int, 1> tensor_m_3
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
    scalar_t* c_ptr = tensor_c.data;
    int* i_ptr = tensor_i.data;
    int* j_ptr = tensor_j.data;
    int* m_1_ptr = tensor_m_1.data;
    int* m_2_ptr = tensor_m_2.data;
    int* m_3_ptr = tensor_m_3.data;

    size_t E = tensor_i.shape[0];
    size_t N = tensor_c.shape[0];
    size_t size_a = tensor_a.shape[1];
    size_t size_r = tensor_r.shape[1];
    size_t o_shift_first_dim = output.shape[1] * output.shape[2];
    size_t x_shift_first_dim = tensor_x.shape[1] * tensor_x.shape[2];
    size_t o_shift_second_dim = output.shape[2];
    size_t x_shift_second_dim = tensor_x.shape[2];

    std::vector<int32_t> first_occurrences = find_first_occurrences(i_ptr, E, output.shape[0]);
    std::fill(o_ptr, o_ptr+output.shape[0]*o_shift_first_dim, static_cast<scalar_t>(0.0));

    #pragma omp parallel for 
    for (size_t i_position = 0; i_position < output.shape[0]; i_position++) {
        scalar_t* o_ptr_shifted_first_dim = o_ptr + i_position * o_shift_first_dim;
        int32_t index_j_start = first_occurrences[i_position];
        int32_t index_j_end = first_occurrences[i_position+1];
        for (int32_t e = index_j_start; e < index_j_end; e++) {
            scalar_t* a_ptr_shifted_first_dim = a_ptr + e * size_a;
            scalar_t* r_ptr_shifted_first_dim = r_ptr + e * size_r;
            scalar_t* x_ptr_shifted_first_dim = x_ptr + j_ptr[e] * x_shift_first_dim;
            for (size_t n = 0; n < N; n++) {
                scalar_t current_c = c_ptr[n];
                scalar_t current_a = a_ptr_shifted_first_dim[m_1_ptr[n]];
                scalar_t current_ac = current_c * current_a;
                scalar_t* o_ptr_shifted_second_dim = o_ptr_shifted_first_dim + m_3_ptr[n] * o_shift_second_dim;
                scalar_t* x_ptr_shifted_second_dim = x_ptr_shifted_first_dim + m_2_ptr[n] * x_shift_second_dim;
                for (size_t r_idx = 0; r_idx < size_r; r_idx++) {
                    o_ptr_shifted_second_dim[r_idx] += current_ac * r_ptr_shifted_first_dim[r_idx] * x_ptr_shifted_second_dim[r_idx];
                }
            }
        }
    }
}
