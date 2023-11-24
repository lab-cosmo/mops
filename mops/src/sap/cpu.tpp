#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sap.hpp"

template<typename scalar_t>
void mops::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    if (A.shape[0] != B.shape[0]) {
        throw std::runtime_error(
            "A and B tensors must have the same number of elements along the "
            "first dimension, got " + std::to_string(A.shape[0]) + " and " +
            std::to_string(B.shape[0])
        );
    }

    if (A.shape[0] != output.shape[0]) {
        throw std::runtime_error(
            "O must contain the same number of elements as the first "
            "dimension of A and B , got " + std::to_string(output.shape[0]) +
            " and " + std::to_string(A.shape[0])
        );
    }

    if (C.shape[0] != indices_B.shape[0]) {
        throw std::runtime_error(
            "the dimension of C must match that of indices_B, got "
            + std::to_string(C.shape[0]) +
            " for C and " + std::to_string(indices_B.shape[0]) + " for indices_B"
        );
    }

    if (C.shape[0] != indices_output.shape[0]) {
        throw std::runtime_error(
            "the dimension of C must match that of indices_output, got "
            + std::to_string(C.shape[0]) +
            " for C and " + std::to_string(indices_output.shape[0]) + " for indices_output"
        );
    }

    // TODO: check sorting (not necessary here, necessary in CUDA implementation)?

    scalar_t* a_ptr = A.data;
    scalar_t* b_ptr = B.data;
    scalar_t* o_ptr = output.data;
    scalar_t* c_ptr = C.data;
    int32_t* p_a_ptr = indices_A.data;
    int32_t* p_b_ptr = indices_B.data;
    int32_t* p_o_ptr = indices_output.data;

    size_t size_first_dimension = A.shape[0];
    size_t size_second_dimension_a = A.shape[1];
    size_t size_second_dimension_b = B.shape[1];
    size_t size_second_dimension_o = output.shape[1];
    size_t c_size = C.shape[0];

    for (int i = 0; i < size_first_dimension; i++) {
        size_t shift_first_dimension_a = i * size_second_dimension_a;
        size_t shift_first_dimension_b = i * size_second_dimension_b;
        size_t shift_first_dimension_o = i * size_second_dimension_o;
        for (int j = 0; j < c_size; j++) {
            o_ptr[shift_first_dimension_o + p_o_ptr[j]] +=
            c_ptr[j] * a_ptr[shift_first_dimension_a + p_a_ptr[j]] * b_ptr[shift_first_dimension_b + p_b_ptr[j]];
        }
    }
}


template<typename scalar_t>
void mops::sparse_accumulation_of_products_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    // TODO: checks

    bool calculate_grad_A = grad_A.data != nullptr;
    bool calculate_grad_B = grad_B.data != nullptr;

    if (calculate_grad_A || calculate_grad_B) {

        scalar_t* a_ptr = A.data;
        scalar_t* b_ptr = B.data;
        scalar_t* grad_o_ptr = grad_output.data;
        scalar_t* c_ptr = C.data;
        int32_t* p_a_ptr = indices_A.data;
        int32_t* p_b_ptr = indices_B.data;
        int32_t* p_o_ptr = indices_output.data;
        scalar_t* grad_a_ptr = grad_A.data;
        scalar_t* grad_b_ptr = grad_B.data;

        size_t size_first_dimension = A.shape[0];
        size_t size_second_dimension_a = A.shape[1];
        size_t size_second_dimension_b = B.shape[1];
        size_t size_second_dimension_o = grad_output.shape[1];
        size_t c_size = C.shape[0];

        scalar_t* grad_output_row = grad_o_ptr; 
        scalar_t* grad_a_row = grad_a_ptr;
        scalar_t* grad_b_row = grad_b_ptr;
        scalar_t* a_row = a_ptr;
        scalar_t* b_row = b_ptr;
        for (int i = 0; i < size_first_dimension; i++){
            for (int j = 0; j < c_size; j++) {                
                scalar_t grad_output_j = grad_output_row[p_o_ptr[j]];
                scalar_t common_factor = grad_output_j * c_ptr[j];
                // These will have to be if constexpr
                if (calculate_grad_A) {
                    grad_a_row[p_a_ptr[j]] += common_factor * b_row[p_b_ptr[j]];
                }
                if (calculate_grad_B) {
                    grad_b_row[p_b_ptr[j]] += common_factor * a_row[p_a_ptr[j]];
                }
            }
            grad_output_row += size_second_dimension_o;
            grad_a_row += size_second_dimension_a;
            grad_b_row += size_second_dimension_b;
            a_row += size_second_dimension_a;
            b_row += size_second_dimension_b;
        }
    }

}
