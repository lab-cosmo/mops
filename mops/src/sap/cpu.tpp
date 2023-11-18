#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sap.hpp"
#include "mops/checks.hpp"

template<typename scalar_t>
void mops::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_b,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int32_t, 1> p_a,
    Tensor<int32_t, 1> p_b,
    Tensor<int32_t, 1> p_o
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
        
    scalar_t* a_ptr = tensor_a.data;
    scalar_t* b_ptr = tensor_b.data;
    scalar_t* o_ptr = output.data;
    scalar_t* c_ptr = tensor_c.data;
    int32_t* p_a_ptr = p_a.data;
    int32_t* p_b_ptr = p_b.data;
    int32_t* p_o_ptr = p_o.data;

    size_t size_first_dimension = tensor_a.shape[0];
    size_t size_second_dimension_a = tensor_a.shape[1];
    size_t size_second_dimension_b = tensor_b.shape[1];
    size_t size_second_dimension_o = output.shape[1];
    size_t c_size = tensor_c.shape[0];

    std::fill(o_ptr, o_ptr+size_first_dimension*output.shape[1], static_cast<scalar_t>(0.0));
    
    for (size_t i = 0; i < size_first_dimension; i++) {
        size_t shift_first_dimension_a = i * size_second_dimension_a;
        size_t shift_first_dimension_b = i * size_second_dimension_b;
        size_t shift_first_dimension_o = i * size_second_dimension_o;
        for (size_t j = 0; j < c_size; j++) { 
            o_ptr[shift_first_dimension_o + p_o_ptr[j]] +=
            c_ptr[j] * a_ptr[shift_first_dimension_a + p_a_ptr[j]] * b_ptr[shift_first_dimension_b + p_b_ptr[j]];                                             
        }
    }
}
