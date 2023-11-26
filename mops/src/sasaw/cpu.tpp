#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sasaw.hpp"
#include "mops/checks.hpp"

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
    check_sizes(A, "A", 0, B, "B", 0, "sasaw");
    check_sizes(W, "W", 0, output, "O", 0, "sasaw");
    check_sizes(B, "B", 1, W, "W", 2, "sasaw");
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, "sasaw");
    check_sizes(C, "C", 0, indices_W_2, "indices_W_2", 0, "sasaw");
    check_sizes(C, "C", 0, indices_output_2, "indices_output_2", 0, "sasaw");
    check_sizes(A, "A", 0, indices_output_1, "indices_output_1", 0, "sasaw");
    check_sizes(A, "A", 0, indices_W_1, "indices_W_1", 0, "sasaw");
    check_index_tensor(indices_output_1, "indices_output_1", output.shape[0], "sasaw");
    check_index_tensor(indices_W_1, "indices_W_1", output.shape[0], "sasaw");
    check_index_tensor(indices_A, "indices_A", A.shape[1], "sasaw");
    check_index_tensor(indices_W_2, "indices_W_2", B.shape[1], "sasaw");
    check_index_tensor(indices_output_2, "indices_output_2", output.shape[1], "sasaw");

    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = A.data;
    scalar_t* r_ptr = B.data;
    scalar_t* x_ptr = W.data;
    scalar_t* c_ptr = C.data;
    int* i_ptr = indices_output_1.data;
    int* j_ptr = indices_W_1.data;
    int* m_1_ptr = indices_A.data;
    int* m_2_ptr = indices_W_2.data;
    int* m_3_ptr = indices_output_2.data;

    size_t E = indices_output_1.shape[0];
    size_t N = C.shape[0];
    size_t size_a = A.shape[1];
    size_t size_r = B.shape[1];
    size_t o_shift_first_dim = output.shape[1] * output.shape[2];
    size_t x_shift_first_dim = W.shape[1] * W.shape[2];
    size_t o_shift_second_dim = output.shape[2];
    size_t x_shift_second_dim = W.shape[2];

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


template<typename scalar_t>
void mops::sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_W,
    Tensor<scalar_t, 3> grad_output,
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
    bool calculate_grad_A = grad_A.data != nullptr;
    bool calculate_grad_B = grad_B.data != nullptr;
    bool calculate_grad_W = grad_W.data != nullptr;

    if (calculate_grad_A || calculate_grad_B || calculate_grad_W) {

        scalar_t* grad_a_ptr = grad_A.data;
        scalar_t* grad_b_ptr = grad_B.data;
        scalar_t* grad_w_ptr = grad_W.data;
        scalar_t* grad_o_ptr = grad_output.data;
        scalar_t* a_ptr = A.data;
        scalar_t* r_ptr = B.data;
        scalar_t* x_ptr = W.data;
        scalar_t* c_ptr = C.data;
        int* i_ptr = indices_output_1.data;
        int* j_ptr = indices_W_1.data;
        int* m_1_ptr = indices_A.data;
        int* m_2_ptr = indices_W_2.data;
        int* m_3_ptr = indices_output_2.data;

        size_t E = indices_output_1.shape[0];
        size_t N = C.shape[0];
        size_t size_a = A.shape[1];
        size_t size_r = B.shape[1];
        size_t o_shift_first_dim = grad_output.shape[1] * grad_output.shape[2];
        size_t x_shift_first_dim = W.shape[1] * W.shape[2];
        size_t o_shift_second_dim = grad_output.shape[2];
        size_t x_shift_second_dim = W.shape[2];

        // For now, we assume that grad_A, grad_B, grad_W are zero-initialized
        // std::fill(grad_a_ptr, grad_a_ptr+A.shape[0]*A.shape[1], static_cast<scalar_t>(0.0));
        // std::fill(grad_b_ptr, grad_b_ptr+B.shape[0]*B.shape[1], static_cast<scalar_t>(0.0));
        // std::fill(grad_w_ptr, grad_w_ptr+W.shape[0]*W.shape[1], static_cast<scalar_t>(0.0));

        for (size_t e = 0; e < E; e++) {
            scalar_t* grad_o_ptr_shifted_first_dim = grad_o_ptr + i_ptr[e] * o_shift_first_dim;
            scalar_t* a_ptr_shifted_first_dim = a_ptr + e * size_a;
            scalar_t* r_ptr_shifted_first_dim = r_ptr + e * size_r;
            scalar_t* x_ptr_shifted_first_dim = x_ptr + j_ptr[e] * x_shift_first_dim;
            for (size_t n = 0; n < N; n++) {
                scalar_t current_c = c_ptr[n];
                scalar_t current_a = a_ptr_shifted_first_dim[m_1_ptr[n]];
                scalar_t current_ac = current_c * current_a;
                scalar_t* grad_o_ptr_shifted_second_dim = grad_o_ptr_shifted_first_dim + m_3_ptr[n] * o_shift_second_dim;
                scalar_t* x_ptr_shifted_second_dim = x_ptr_shifted_first_dim + m_2_ptr[n] * x_shift_second_dim;
                for (size_t r_idx = 0; r_idx < size_r; r_idx++) {
                    scalar_t current_grad_o = grad_o_ptr_shifted_second_dim[r_idx];
                    if (calculate_grad_A) {
                        grad_a_ptr[e * size_a + m_1_ptr[n]] += current_grad_o * current_c * r_ptr_shifted_first_dim[r_idx] * x_ptr_shifted_second_dim[r_idx];
                    }
                    if (calculate_grad_B) {
                        grad_b_ptr[e * size_r + r_idx] += current_grad_o * current_ac * x_ptr_shifted_second_dim[r_idx];
                    }
                    if (calculate_grad_W) {
                        grad_w_ptr[j_ptr[e] * x_shift_first_dim + m_2_ptr[n] * x_shift_second_dim + r_idx] += current_grad_o * current_ac * r_ptr_shifted_first_dim[r_idx];
                    }
                }
            }
        }

    }
}
