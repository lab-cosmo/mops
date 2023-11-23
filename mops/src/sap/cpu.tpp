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
    check_sizes(tensor_a, "A", 0, tensor_b, "B", 0, "sap");
    check_sizes(tensor_a, "A", 0, output, "O", 0, "sap");
    check_sizes(tensor_c, "C", 0, p_a, "P_A", 0, "sap");
    check_sizes(tensor_c, "C", 0, p_b, "P_B", 0, "sap");
    check_sizes(tensor_c, "C", 0, p_o, "P_O", 0, "sap");
    check_index_tensor(p_a, "P_A", tensor_a.shape[1], "sap");
    check_index_tensor(p_b, "P_B", tensor_b.shape[1], "sap");
    check_index_tensor(p_o, "P_O", output.shape[1], "sap");

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
