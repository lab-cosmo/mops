#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsaw.hpp"

template<typename scalar_t>
void mops::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
) {
    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = A.data;
    scalar_t* r_ptr = B.data;
    scalar_t* x_ptr = W.data;
    int* i_ptr = indices_output.data;
    int* j_ptr = indices_W.data;

    size_t E = indices_W.shape[0];
    size_t size_a = A.shape[1];
    size_t size_r = B.shape[1];

    std::fill(o_ptr, o_ptr+output.shape[0]*output.shape[1]*output.shape[2], static_cast<scalar_t>(0.0));

    for (size_t e = 0; e < E; e++) {
        scalar_t* o_ptr_shifted_first_dim = o_ptr + i_ptr[e] * size_a * size_r;
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


template<typename scalar_t>
void mops::outer_product_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
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
        int* i_ptr = indices_output.data;
        int* j_ptr = indices_W.data;

        size_t E = indices_W.shape[0];
        size_t size_a = A.shape[1];
        size_t size_r = B.shape[1];

        // TODO: checks

        // For now, we assume that grad_A, grad_B, grad_W are zero-initialized
        // std::fill(grad_a_ptr, grad_a_ptr+A.shape[0]*A.shape[1], static_cast<scalar_t>(0.0));
        // std::fill(grad_b_ptr, grad_b_ptr+B.shape[0]*B.shape[1], static_cast<scalar_t>(0.0));
        // std::fill(grad_w_ptr, grad_w_ptr+W.shape[0]*W.shape[1], static_cast<scalar_t>(0.0));

        for (size_t e = 0; e < E; e++) {
            scalar_t* grad_o_ptr_shifted_first_dim = grad_o_ptr + i_ptr[e] * size_a * size_r;
            scalar_t* a_ptr_shifted_first_dim = a_ptr + e * size_a;
            scalar_t* r_ptr_shifted_first_dim = r_ptr + e * size_r;
            scalar_t* x_ptr_shifted_first_dim = x_ptr + j_ptr[e] * size_r;
            for (size_t a_idx = 0; a_idx < size_a; a_idx++) {
                scalar_t current_a = a_ptr_shifted_first_dim[a_idx];
                scalar_t* grad_o_ptr_shifted_second_dim = grad_o_ptr_shifted_first_dim + a_idx * size_r;
                for (size_t r_idx = 0; r_idx < size_r; r_idx++) {
                    scalar_t current_grad_o = grad_o_ptr_shifted_second_dim[r_idx];
                    if (calculate_grad_A) {
                        grad_a_ptr[e * size_a + a_idx] += current_grad_o * r_ptr_shifted_first_dim[r_idx] * x_ptr_shifted_first_dim[r_idx];
                    }
                    if (calculate_grad_B) {
                        grad_b_ptr[e * size_r + r_idx] += current_grad_o * current_a * x_ptr_shifted_first_dim[r_idx];
                    }
                    if (calculate_grad_W) {
                        grad_w_ptr[j_ptr[e] * size_r + r_idx] += current_grad_o * current_a * r_ptr_shifted_first_dim[r_idx];
                    }
                }
            }
        }

    }
}
