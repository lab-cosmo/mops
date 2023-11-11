#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sap.hpp"

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
    if (tensor_a.shape[0] != tensor_b.shape[0]) {
        throw std::runtime_error(
            "A and B tensors must have the same number of elements along the "
            "first dimension, got " + std::to_string(tensor_a.shape[0]) + " and " +
            std::to_string(tensor_b.shape[0])
        );
    }

    if (tensor_a.shape[0] != output.shape[0]) {
        throw std::runtime_error(
            "O must contain the same number of elements as the first "
            "dimension of A and B , got " + std::to_string(output.shape[0]) +
            " and " + std::to_string(tensor_a.shape[0])
        );
    }

    if (tensor_c.shape[0] != p_b.shape[0]) {
        throw std::runtime_error(
            "the dimension of C must match that of P_B, got "
            + std::to_string(tensor_c.shape[0]) +
            " for C and " + std::to_string(p_b.shape[0]) + " for P_B"
        );
    }

    if (tensor_c.shape[0] != p_o.shape[0]) {
        throw std::runtime_error(
            "the dimension of C must match that of P_O, got "
            + std::to_string(tensor_c.shape[0]) +
            " for C and " + std::to_string(p_o.shape[0]) + " for P_O"
        );
    }

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
    
    for (size_t i = 0; i < size_first_dimension; i++) {
        size_tshift_first_dimension_a = i * size_second_dimension_a;
        size_tshift_first_dimension_b = i * size_second_dimension_b;
        size_tshift_first_dimension_o = i * size_second_dimension_o;
        for (size_t j = 0; j < c_size; j++) { 
            o_ptr[shift_first_dimension_o + p_o_ptr[j]] +=
            c_ptr[j] * a_ptr[shift_first_dimension_a + p_a_ptr[j]] * b_ptr[shift_first_dimension_b + p_b_ptr[j]];                                             
        }
    }
}
