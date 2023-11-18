#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sasax.hpp"
#include "mops/checks.hpp"

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
    check_sizes(tensor_a, "A", 0, tensor_r, "R", 0, "sasax");
    check_sizes(tensor_x, "X", 0, output, "O", 0, "sasax");
    check_sizes(tensor_r, "R", 1, tensor_x, "X", 2, "sasax");
    check_sizes(tensor_c, "C", 0, tensor_m_1, "M_1", 0, "sasax");
    check_sizes(tensor_c, "C", 0, tensor_m_2, "M_2", 0, "sasax");
    check_sizes(tensor_c, "C", 0, tensor_m_3, "M_3", 0, "sasax");
    check_sizes(tensor_a, "A", 0, tensor_i, "I", 0, "sasax");
    check_sizes(tensor_a, "A", 0, tensor_j, "J", 0, "sasax");
    check_index_tensor(tensor_i, "I", output.shape[0], "sasax");
    check_index_tensor(tensor_j, "J", output.shape[0], "sasax");
    check_index_tensor(tensor_m_1, "M_1", tensor_a.shape[1], "sasax");
    check_index_tensor(tensor_m_2, "M_2", tensor_r.shape[1], "sasax");
    check_index_tensor(tensor_m_3, "M_3", output.shape[1], "sasax");

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
