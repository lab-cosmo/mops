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

    long size_first_dimension = A.shape[0];
    long size_second_dimension_a = A.shape[1];
    long size_second_dimension_b = B.shape[1];
    long size_second_dimension_o = output.shape[1];
    long c_size = C.shape[0];

    for (int i = 0; i < size_first_dimension; i++) {
        long shift_first_dimension_a = i * size_second_dimension_a;
        long shift_first_dimension_b = i * size_second_dimension_b;
        long shift_first_dimension_o = i * size_second_dimension_o;
        for (int j = 0; j < c_size; j++) {
            o_ptr[shift_first_dimension_o + p_o_ptr[j]] +=
            c_ptr[j] * a_ptr[shift_first_dimension_a + p_a_ptr[j]] * b_ptr[shift_first_dimension_b + p_b_ptr[j]];
        }
    }
}
