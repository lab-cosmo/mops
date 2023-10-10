#ifndef MOPS_EXAMPLE_CAPI_H
#define MOPS_EXAMPLE_CAPI_H

#include <stddef.h>
#include "mops/exports.h"


#ifdef __cplusplus
extern "C" {
#endif

void MOPS_EXPORT mops_example_values_f32(
    const float* input,
    size_t input_length,
    float* output,
    size_t output_length
);

void MOPS_EXPORT mops_example_values_f64(
    const double* input,
    size_t input_length,
    double* output,
    size_t output_length
);

void MOPS_EXPORT mops_example_jvp_f32(
    const float* grad_output,
    size_t grad_output_length,
    float* grad_input,
    size_t grad_input_length
);

void MOPS_EXPORT mops_example_jvp_f64(
    const double* grad_output,
    size_t grad_output_length,
    double* grad_input,
    size_t grad_input_length
);


#ifdef __cplusplus
}
#endif


#endif
