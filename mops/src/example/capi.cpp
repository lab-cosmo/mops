#include "mops/example.hpp"
#include "mops/example.h"

extern "C" void mops_example_values_f32(
    const float* input,
    size_t input_length,
    float* output,
    size_t output_length
) {
    return mops::example_values(input, input_length, output, output_length);
}

extern "C" void mops_example_values_f64(
    const double* input,
    size_t input_length,
    double* output,
    size_t output_length
) {
    return mops::example_values(input, input_length, output, output_length);
}


extern "C" void mops_example_jvp_f32(
    const float* grad_output,
    size_t grad_output_length,
    float* grad_input,
    size_t grad_input_length
) {
    return mops::example_jvp(grad_output, grad_output_length, grad_input, grad_input_length);
}

extern "C" void mops_example_jvp_f64(
    const double* grad_output,
    size_t grad_output_length,
    double* grad_input,
    size_t grad_input_length
) {
    return mops::example_jvp(grad_output, grad_output_length, grad_input, grad_input_length);
}
