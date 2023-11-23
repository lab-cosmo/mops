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

    std::fill(o_ptr, o_ptr+output.shape[0]*o_shift_first_dim, static_cast<scalar_t>(0.0));

    for (size_t e = 0; e < E; e++) {
        scalar_t* o_ptr_shifted_first_dim = o_ptr + i_ptr[e] * o_shift_first_dim;
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
