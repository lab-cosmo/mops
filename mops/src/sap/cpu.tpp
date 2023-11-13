#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sap.hpp"


template<typename scalar_t>
constexpr size_t get_simd_element_count();
// Assume 256-bit vector registers. A conservative choice.
template<>
constexpr size_t get_simd_element_count<double>() {
    return 256 / (sizeof(double) * 8);  // 256 bits / 64 bits
}
template<>
constexpr size_t get_simd_element_count<float>() {
    return 256 / (sizeof(float) * 8);   // 256 bits / 32 bits
}

template<typename scalar_t, size_t simd_element_count>
void interleave_tensor(
    mops::Tensor<scalar_t, 2> initial_data,
    scalar_t* interleft_data,
    scalar_t* remainder_data
) {

    size_t batch_dim = initial_data.shape[0];
    size_t remainder = batch_dim % simd_element_count;
    size_t quotient = batch_dim / simd_element_count;
    size_t calculation_dim = initial_data.shape[1];
    scalar_t* initial_data_ptr = initial_data.data;

    for (size_t i=0; i<quotient; i++) {
        for (size_t j=0; j<calculation_dim; j++) {
            for (size_t k=0; k<simd_element_count; k++) {
                interleft_data[i*calculation_dim*simd_element_count+j*simd_element_count+k] = initial_data_ptr[i*simd_element_count*calculation_dim+k*calculation_dim+j];
            }
        }
    }

    for (size_t j=0; j<calculation_dim; j++) {
        for (size_t k=0; k<remainder; k++) {
            remainder_data[j*remainder+k] = initial_data_ptr[quotient*simd_element_count*calculation_dim+k*calculation_dim+j];
        }
    }
}

template<typename scalar_t, size_t simd_element_count>
void un_interleave_tensor(
    mops::Tensor<scalar_t, 2> output_data,
    scalar_t* interleft_data,
    scalar_t* remainder_data
) {

    size_t batch_dim = output_data.shape[0];
    size_t remainder = batch_dim % simd_element_count;
    size_t quotient = batch_dim / simd_element_count;
    size_t calculation_dim = output_data.shape[1];
    scalar_t* output_data_ptr = output_data.data;

    for (size_t i=0; i<quotient; i++) {
        for (size_t j=0; j<calculation_dim; j++) {
            for (size_t k=0; k<simd_element_count; k++) {
                output_data_ptr[i*simd_element_count*calculation_dim+k*calculation_dim+j] = interleft_data[i*calculation_dim*simd_element_count+j*simd_element_count+k];
            }
        }
    }

    for (size_t j=0; j<calculation_dim; j++) {
        for (size_t k=0; k<remainder; k++) {
            output_data_ptr[quotient*simd_element_count*calculation_dim+k*calculation_dim+j] = remainder_data[j*remainder+k];
        }
    }
}

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
        
    // scalar_t* a_ptr = tensor_a.data;
    // scalar_t* b_ptr = tensor_b.data;
    // scalar_t* o_ptr = output.data;
    scalar_t* c_ptr = tensor_c.data;
    int32_t* p_a_ptr = p_a.data;
    int32_t* p_b_ptr = p_b.data;
    int32_t* p_o_ptr = p_o.data;

    size_t size_first_dimension = tensor_a.shape[0];
    size_t size_second_dimension_a = tensor_a.shape[1];
    size_t size_second_dimension_b = tensor_b.shape[1];
    size_t size_second_dimension_o = output.shape[1];
    size_t c_size = tensor_c.shape[0];

    constexpr size_t simd_element_count = get_simd_element_count<scalar_t>();

    size_t size_first_dimension_interleft = size_first_dimension/simd_element_count;
    size_t size_remainder = size_first_dimension%simd_element_count;

    scalar_t* interleft_o_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_o*simd_element_count];
    scalar_t* remainder_o_ptr = new scalar_t[size_remainder*size_second_dimension_o];

    scalar_t* interleft_a_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_a*simd_element_count];
    scalar_t* remainder_a_ptr = new scalar_t[size_remainder*size_second_dimension_a];
    interleave_tensor<scalar_t, simd_element_count>(tensor_a, interleft_a_ptr, remainder_a_ptr);

    scalar_t* interleft_b_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_b*simd_element_count];
    scalar_t* remainder_b_ptr = new scalar_t[size_remainder*size_second_dimension_b];
    interleave_tensor<scalar_t, simd_element_count>(tensor_b, interleft_b_ptr, remainder_b_ptr);

    std::fill(interleft_o_ptr, interleft_o_ptr+size_first_dimension_interleft*size_second_dimension_o*simd_element_count, static_cast<scalar_t>(0.0));
    std::fill(remainder_o_ptr, remainder_o_ptr+size_remainder*size_second_dimension_o, static_cast<scalar_t>(0.0));
    
    for (size_t i = 0; i < size_first_dimension_interleft; i++) {
        scalar_t* a_ptr_shifted_first_dim = interleft_a_ptr + i * size_second_dimension_a*simd_element_count;
        scalar_t* b_ptr_shifted_first_dim = interleft_b_ptr + i * size_second_dimension_b*simd_element_count;
        scalar_t* o_ptr_shifted_first_dim = interleft_o_ptr + i * size_second_dimension_o*simd_element_count;
        for (size_t j = 0; j < c_size; j++) {
            scalar_t* a_ptr_shifted_second_dim = a_ptr_shifted_first_dim + p_a_ptr[j] * simd_element_count;
            scalar_t* b_ptr_shifted_second_dim = b_ptr_shifted_first_dim + p_b_ptr[j] * simd_element_count;
            scalar_t* o_ptr_shifted_second_dim = o_ptr_shifted_first_dim + p_o_ptr[j] * simd_element_count;
            scalar_t current_c = c_ptr[j];
            for (size_t k = 0; k < simd_element_count; k++) {
                o_ptr_shifted_second_dim[k] += current_c * a_ptr_shifted_second_dim[k] * b_ptr_shifted_second_dim[k];
            }                                             
        }
    }

    // Handle remainder
    for (size_t j = 0; j < c_size; j++) {
        scalar_t* a_ptr_shifted_second_dim = remainder_a_ptr + p_a_ptr[j] * size_remainder;
        scalar_t* b_ptr_shifted_second_dim = remainder_b_ptr + p_b_ptr[j] * size_remainder;
        scalar_t* o_ptr_shifted_second_dim = remainder_o_ptr + p_o_ptr[j] * size_remainder;
        scalar_t current_c = c_ptr[j];
        for (size_t k = 0; k < size_remainder; k++) {
            o_ptr_shifted_second_dim[k] += current_c * a_ptr_shifted_second_dim[k] * b_ptr_shifted_second_dim[k];
        }                                             
    }

    un_interleave_tensor<scalar_t, simd_element_count>(output, interleft_o_ptr, remainder_o_ptr);

    delete[] interleft_o_ptr;
    delete[] interleft_a_ptr;
    delete[] interleft_b_ptr;
    delete[] remainder_o_ptr;
    delete[] remainder_a_ptr;
    delete[] remainder_b_ptr;
}
