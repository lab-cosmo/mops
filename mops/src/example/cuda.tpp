#include "mops/example.hpp"

template<typename scalar_t>
void mops::cuda::example_values(
    int cuda_device,
    const scalar_t* input,
    size_t input_length,
    scalar_t* output,
    size_t output_length
) {
    // todo: cuda host code
}

template<typename scalar_t>
void mops::cuda::example_jvp(
    int cuda_device,
    const scalar_t* grad_output,
    size_t grad_output_length,
    scalar_t* grad_input,
    size_t grad_input_length
) {
    // todo: cuda host code
}
