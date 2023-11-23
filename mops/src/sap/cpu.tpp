#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sap.hpp"
#include "mops/checks.hpp"

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
    check_sizes(A, "A", 0, B, "B", 0, "sap");
    check_sizes(A, "A", 0, output, "O", 0, "sap");
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, "sap");
    check_sizes(C, "C", 0, indices_B, "indices_B", 0, "sap");
    check_sizes(C, "C", 0, indices_output, "indices_output", 0, "sap");
    check_index_tensor(indices_A, "indices_A", A.shape[1], "sap");
    check_index_tensor(indices_B, "indices_B", B.shape[1], "sap");
    check_index_tensor(indices_output, "indices_output", output.shape[1], "sap");

    // TODO: check sorting (not necessary here, necessary in CUDA implementation)?

    scalar_t* a_ptr = A.data;
    scalar_t* b_ptr = B.data;
    scalar_t* o_ptr = output.data;
    scalar_t* c_ptr = C.data;
    int32_t* indices_A_ptr = indices_A.data;
    int32_t* indices_B_ptr = indices_B.data;
    int32_t* indices_output_ptr = indices_output.data;

    size_t size_first_dimension = A.shape[0];
    size_t size_second_dimension_a = A.shape[1];
    size_t size_second_dimension_b = B.shape[1];
    size_t size_second_dimension_o = output.shape[1];
    size_t c_size = C.shape[0];

    std::fill(o_ptr, o_ptr+output.shape[0]*output.shape[1], static_cast<scalar_t>(0.0));

    for (size_t i = 0; i < size_first_dimension; i++) {
        size_t shift_first_dimension_a = i * size_second_dimension_a;
        size_t shift_first_dimension_b = i * size_second_dimension_b;
        size_t shift_first_dimension_o = i * size_second_dimension_o;
        for (size_t j = 0; j < c_size; j++) {
            o_ptr[shift_first_dimension_o + indices_output_ptr[j]] +=
            c_ptr[j] * a_ptr[shift_first_dimension_a + indices_A_ptr[j]] * b_ptr[shift_first_dimension_b + indices_B_ptr[j]];
        }
    }
}
