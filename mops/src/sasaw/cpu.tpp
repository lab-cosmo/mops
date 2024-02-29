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
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
) {
    check_sizes(A, "A", 0, B, "B", 0, "sasaw");
    check_sizes(W, "W", 0, output, "output", 0, "sasaw");
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
    scalar_t* b_ptr = B.data;
    scalar_t* w_ptr = W.data;
    scalar_t* c_ptr = C.data;

    const int32_t* idx_o1_ptr = indices_output_1.data;
    const int32_t* idx_w1_ptr = indices_W_1.data;
    const int32_t* idx_a_ptr = indices_A.data;
    const int32_t* idx_w2_ptr = indices_W_2.data;
    const int32_t* idx_o2_ptr = indices_output_2.data;

    size_t N = C.shape[0];
    size_t size_a = A.shape[1];
    size_t size_b = B.shape[1];
    size_t size_output_first_dim = output.shape[0];
    size_t output_shift_first_dim = output.shape[1] * output.shape[2];
    size_t w_shift_first_dim = W.shape[1] * W.shape[2];
    size_t output_shift_second_dim = output.shape[2];
    size_t w_shift_second_dim = W.shape[2];

    // For each index in the first dimension of the outputs,
    // get what indices in the inputs should write to it
    std::vector<std::vector<size_t>> write_list = get_write_list(indices_output_1);
    std::fill(o_ptr, o_ptr+output.shape[0]*output_shift_first_dim, static_cast<scalar_t>(0.0));

    #pragma omp parallel for
    for (size_t i = 0; i < size_output_first_dim; i++) {
        // iterate over input indices that will write to the output index i
        // these are stored as entries (variable "e") in write_list[i]
        // for example, `o_ptr_e` is a pointer to the first element of 
        // the row in the O tensor where the current contributions will be added
        for (size_t e : write_list[i]) {
            scalar_t* o_ptr_e = o_ptr + idx_o1_ptr[e] * output_shift_first_dim;
            scalar_t* a_ptr_e = a_ptr + e * size_a;
            scalar_t* b_ptr_e = b_ptr + e * size_b;
            scalar_t* w_ptr_e = w_ptr + idx_w1_ptr[e] * w_shift_first_dim;
            for (size_t n = 0; n < N; n++) {
                scalar_t current_c = c_ptr[n];
                scalar_t current_a = a_ptr_e[idx_a_ptr[n]];
                scalar_t current_ac = current_c * current_a;
                scalar_t* o_ptr_e_n = o_ptr_e + idx_o2_ptr[n] * output_shift_second_dim;
                scalar_t* w_ptr_e_n = w_ptr_e + idx_w2_ptr[n] * w_shift_second_dim;
                for (size_t b_idx = 0; b_idx < size_b; b_idx++) {
                    o_ptr_e_n[b_idx] += current_ac * b_ptr_e[b_idx] * w_ptr_e_n[b_idx];
                }
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
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
) {
    check_sizes(A, "A", 0, B, "B", 0, "sasaw_vjp");
    check_sizes(W, "W", 0, grad_output, "grad_output", 0, "sasaw_vjp");
    check_sizes(B, "B", 1, W, "W", 2, "sasaw_vjp");
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, "sasaw_vjp");
    check_sizes(C, "C", 0, indices_W_2, "indices_W_2", 0, "sasaw_vjp");
    check_sizes(C, "C", 0, indices_output_2, "indices_output_2", 0, "sasaw_vjp");
    check_sizes(A, "A", 0, indices_output_1, "indices_output_1", 0, "sasaw_vjp");
    check_sizes(A, "A", 0, indices_W_1, "indices_W_1", 0, "sasaw_vjp");
    check_index_tensor(indices_output_1, "indices_output_1", grad_output.shape[0], "sasaw_vjp");
    check_index_tensor(indices_W_1, "indices_W_1", grad_output.shape[0], "sasaw_vjp");
    check_index_tensor(indices_A, "indices_A", A.shape[1], "sasaw_vjp");
    check_index_tensor(indices_W_2, "indices_W_2", B.shape[1], "sasaw_vjp");
    check_index_tensor(indices_output_2, "indices_output_2", grad_output.shape[1], "sasaw_vjp");

    bool calculate_grad_A = grad_A.data != nullptr;
    bool calculate_grad_B = grad_B.data != nullptr;
    bool calculate_grad_W = grad_W.data != nullptr;

    if (calculate_grad_A || calculate_grad_B || calculate_grad_W) {

        if (calculate_grad_A) {
            check_same_shape(grad_A, "grad_A", A, "A", "sasaw_vjp");
            std::fill(grad_A.data, grad_A.data+A.shape[0]*A.shape[1], static_cast<scalar_t>(0.0));
        }
        if (calculate_grad_B) {
            check_same_shape(grad_B, "grad_B", B, "B", "sasaw_vjp");
            std::fill(grad_B.data, grad_B.data+B.shape[0]*B.shape[1], static_cast<scalar_t>(0.0));
        }
        if (calculate_grad_W) {
            check_same_shape(grad_W, "grad_W", W, "W", "sasaw_vjp");
            std::fill(grad_W.data, grad_W.data+W.shape[0]*W.shape[1]*W.shape[2], static_cast<scalar_t>(0.0));
        }

        scalar_t* grad_a_ptr = grad_A.data;
        scalar_t* grad_b_ptr = grad_B.data;
        scalar_t* grad_w_ptr = grad_W.data;
        scalar_t* grad_o_ptr = grad_output.data;
        scalar_t* a_ptr = A.data;
        scalar_t* b_ptr = B.data;
        scalar_t* w_ptr = W.data;
        scalar_t* c_ptr = C.data;
        const int32_t* idx_o1_ptr = indices_output_1.data;
        const int32_t* idx_a_ptr = indices_A.data;
        const int32_t* idx_w2_ptr = indices_W_2.data;
        const int32_t* idx_o2_ptr = indices_output_2.data;

        size_t N = C.shape[0];
        size_t size_a = A.shape[1];
        size_t size_b = B.shape[1];
        size_t size_output_first_dim = grad_output.shape[0];
        size_t output_shift_first_dim = grad_output.shape[1] * grad_output.shape[2];
        size_t w_shift_first_dim = W.shape[1] * W.shape[2];
        size_t output_shift_second_dim = grad_output.shape[2];
        size_t w_shift_second_dim = W.shape[2];

        // For each index in the first dimension of grad_W,
        // get what indices in grad_output should write to it
        std::vector<std::vector<size_t>> write_list = get_write_list(indices_W_1);

        #pragma omp parallel for
        for (size_t j = 0; j < size_output_first_dim; j++) {
            // iterate over grad_output indices that will write to index j in the
            // first dimension of grad_W
            // these are stored as entries (variable "e") in write_list[j]
            scalar_t* w_ptr_e = w_ptr + j * w_shift_first_dim;;
            for (size_t e : write_list[j]) {
                scalar_t* grad_o_ptr_e = grad_o_ptr + idx_o1_ptr[e] * output_shift_first_dim;
                scalar_t* a_ptr_e = a_ptr + e * size_a;
                scalar_t* b_ptr_e = b_ptr + e * size_b;
                for (size_t n = 0; n < N; n++) {
                    scalar_t current_c = c_ptr[n];
                    scalar_t current_a = a_ptr_e[idx_a_ptr[n]];
                    scalar_t current_ac = current_c * current_a;
                    scalar_t* grad_o_ptr_e_n = grad_o_ptr_e + idx_o2_ptr[n] * output_shift_second_dim;
                    scalar_t* w_ptr_e_n = w_ptr_e + idx_w2_ptr[n] * w_shift_second_dim;
                    for (size_t b_idx = 0; b_idx < size_b; b_idx++) {
                        scalar_t current_grad_o = grad_o_ptr_e_n[b_idx];
                        if (calculate_grad_A) {
                            grad_a_ptr[e * size_a + idx_a_ptr[n]] += current_grad_o * current_c * b_ptr_e[b_idx] * w_ptr_e_n[b_idx];
                        }
                        if (calculate_grad_B) {
                            grad_b_ptr[e * size_b + b_idx] += current_grad_o * current_ac * w_ptr_e_n[b_idx];
                        }
                        if (calculate_grad_W) {
                            grad_w_ptr[j * w_shift_first_dim + idx_w2_ptr[n] * w_shift_second_dim + b_idx] += current_grad_o * current_ac * b_ptr_e[b_idx];
                        }
                    }
                }
            }
        }

    }
}
