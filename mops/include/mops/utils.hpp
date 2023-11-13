#ifndef UTILS_HPP
#define UTILS_HPP

#include "mops/tensor.hpp"


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
void interleave_tensor(mops::Tensor<scalar_t, 2> initial_data, scalar_t* interleft_data, scalar_t* remainder_data);

template<typename scalar_t, size_t simd_element_count>
void un_interleave_tensor(mops::Tensor<scalar_t, 2> output_data, scalar_t* interleft_data, scalar_t* remainder_data);

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



#endif
