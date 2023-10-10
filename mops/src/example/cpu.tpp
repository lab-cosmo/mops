#include <stdexcept>

#include "mops/example.hpp"

template<typename scalar_t>
void mops::example_values(
    const scalar_t* input,
    size_t input_length,
    scalar_t* output,
    size_t output_length
) {
    // todo: cpu implementation
}

template<typename scalar_t>
void mops::example_jvp(
    const scalar_t* grad_output,
    size_t grad_output_length,
    scalar_t* grad_input,
    size_t grad_input_length
) {
    // todo: cpu implementation
}
