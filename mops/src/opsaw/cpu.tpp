#include <algorithm>

#include "mops/opsaw.hpp"

#include "internal/checks.hpp"
#include "internal/utils.hpp"


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
    check_sizes(A, "A", 1, output, "output", 1, "opsaw");
    check_sizes(B, "B", 1, output, "output", 2, "opsaw");
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, "opsaw");
    check_sizes(A, "A", 0, indices_W, "indices_W", 0, "opsaw");
    check_sizes(W, "W", 0, output, "output", 0, "opsaw");
    check_sizes(B, "B", 1, W, "W", 1, "opsaw");
    check_index_tensor(indices_output, "indices_output", output.shape[0], "opsaw");
    check_index_tensor(indices_W, "indices_W", output.shape[0], "opsaw");

    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = A.data;
    scalar_t* b_ptr = B.data;
    scalar_t* w_ptr = W.data;
    const int32_t* idx_w_ptr = indices_W.data;

    size_t size_output_first_dimension = output.shape[0];
    size_t size_a = A.shape[1];
    size_t size_b = B.shape[1];
    size_t size_ab = size_a * size_b;

    // For each index in the first dimension of the outputs,
    // get what indices in the inputs should write to it
    std::vector<std::vector<size_t>> write_list = get_write_list(indices_output);

    std::fill(o_ptr, o_ptr+output.shape[0]*output.shape[1]*output.shape[2], static_cast<scalar_t>(0.0));

    #pragma omp parallel for
    for (size_t i = 0; i < size_output_first_dimension; i++) {
        scalar_t* o_ptr_e = o_ptr + i * size_ab;
        // iterate over input indices that will write to the output index i
        for (size_t e : write_list[i]) {
            scalar_t* a_ptr_e = a_ptr + e * size_a;
            scalar_t* b_ptr_e = b_ptr + e * size_b;
            scalar_t* w_ptr_e = w_ptr + idx_w_ptr[e] * size_b;
            for (size_t a_idx = 0; a_idx < size_a; a_idx++) {
                scalar_t current_a = a_ptr_e[a_idx];
                scalar_t* o_ptr_e_a = o_ptr_e + a_idx * size_b;
                for (size_t b_idx = 0; b_idx < size_b; b_idx++) {
                    o_ptr_e_a[b_idx] += current_a * b_ptr_e[b_idx] * w_ptr_e[b_idx];
                }
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
    check_sizes(A, "A", 0, B, "B", 0, "opsaw_vjp");
    check_sizes(A, "A", 1, grad_output, "grad_output", 1, "opsaw_vjp");
    check_sizes(B, "B", 1, grad_output, "grad_output", 2, "opsaw_vjp");
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, "opsaw_vjp");
    check_sizes(A, "A", 0, indices_W, "indices_W", 0, "opsaw_vjp");
    check_sizes(W, "W", 0, grad_output, "grad_output", 0, "opsaw_vjp");
    check_sizes(B, "B", 1, W, "W", 1, "opsaw_vjp");
    check_index_tensor(indices_output, "indices_output", grad_output.shape[0], "opsaw_vjp");
    check_index_tensor(indices_W, "indices_W", grad_output.shape[0], "opsaw_vjp");

    bool calculate_grad_A = grad_A.data != nullptr;
    bool calculate_grad_B = grad_B.data != nullptr;
    bool calculate_grad_W = grad_W.data != nullptr;

    if (calculate_grad_A || calculate_grad_B || calculate_grad_W) {

        if (calculate_grad_A) {
            check_same_shape(grad_A, "grad_A", A, "A", "opsaw_vjp");
            std::fill(grad_A.data, grad_A.data+A.shape[0]*A.shape[1], static_cast<scalar_t>(0.0));
        }
        if (calculate_grad_B) {
            check_same_shape(grad_B, "grad_B", B, "B", "opsaw_vjp");
            std::fill(grad_B.data, grad_B.data+B.shape[0]*B.shape[1], static_cast<scalar_t>(0.0));
        }
        if (calculate_grad_W) {
            check_same_shape(grad_W, "grad_W", W, "W", "opsaw_vjp");
            std::fill(grad_W.data, grad_W.data+W.shape[0]*W.shape[1], static_cast<scalar_t>(0.0));
        }

        scalar_t* grad_a_ptr = grad_A.data;
        scalar_t* grad_b_ptr = grad_B.data;
        scalar_t* grad_w_ptr = grad_W.data;
        scalar_t* grad_o_ptr = grad_output.data;
        scalar_t* a_ptr = A.data;
        scalar_t* b_ptr = B.data;
        scalar_t* w_ptr = W.data;
        const int32_t* idx_o_ptr = indices_output.data;

        size_t size_output_first_dimension = grad_output.shape[0];
        size_t size_a = A.shape[1];
        size_t size_b = B.shape[1];

        // For each index in the first dimension of the grad_W,
        // get what indices in the first dimension of grad_output should be write to it
        std::vector<std::vector<size_t>> write_list = get_write_list(indices_W);

        #pragma omp parallel for
        for (size_t j = 0; j < size_output_first_dimension; j++) {
            scalar_t* w_ptr_e = w_ptr + j * size_b;
            // iterate over grad_output indices that will write to the grad_W index j
            for (size_t e : write_list[j]) {
                scalar_t* grad_o_ptr_e = grad_o_ptr + idx_o_ptr[e] * size_a * size_b;
                scalar_t* a_ptr_e = a_ptr + e * size_a;
                scalar_t* b_ptr_e = b_ptr + e * size_b;
                for (size_t a_idx = 0; a_idx < size_a; a_idx++) {
                    scalar_t current_a = a_ptr_e[a_idx];
                    scalar_t* grad_o_ptr_e_a = grad_o_ptr_e + a_idx * size_b;
                    for (size_t b_idx = 0; b_idx < size_b; b_idx++) {
                        scalar_t current_grad_o = grad_o_ptr_e_a[b_idx];
                        if (calculate_grad_A) {
                            grad_a_ptr[e * size_a + a_idx] += current_grad_o * b_ptr_e[b_idx] * w_ptr_e[b_idx];
                        }
                        if (calculate_grad_B) {
                            grad_b_ptr[e * size_b + b_idx] += current_grad_o * current_a * w_ptr_e[b_idx];
                        }
                        if (calculate_grad_W) {
                            grad_w_ptr[j * size_b + b_idx] += current_grad_o * current_a * b_ptr_e[b_idx];
                        }
                    }
                }
            }
        }
    }
}

template<typename scalar_t>
void mops::outer_product_scatter_add_with_weights_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_W_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 2> grad_grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> indices_W,
    Tensor<int32_t, 1> indices_output
) {
    throw std::runtime_error("Not implemented");
}
