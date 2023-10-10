#ifndef MOPS_OUTER_PRODUCT_SCATTER_ADD_H
#define MOPS_OUTER_PRODUCT_SCATTER_ADD_H

#include <stddef.h>
#include <stdint.h>
#include "mops/exports.h"


#ifdef __cplusplus
extern "C" {
#endif

/// CPU version of mops::outer_product_scatter_add for 32-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_f32(
    float* output,
    int64_t output_shape_1,
    int64_t output_shape_2,
    const float* tensor_a,
    int64_t tensor_a_shape_1,
    int64_t tensor_a_shape_2,
    const float* tensor_b,
    int64_t tensor_b_shape_1,
    int64_t tensor_b_shape_2,
    const int32_t* indexes,
    int64_t indexes_shape_1
);


/// CPU version of mops::outer_product_scatter_add for 64-bit floats
int MOPS_EXPORT mops_outer_product_scatter_add_f64(
    double* output,
    int64_t output_shape_1,
    int64_t output_shape_2,
    const double* tensor_a,
    int64_t tensor_a_shape_1,
    int64_t tensor_a_shape_2,
    const double* tensor_b,
    int64_t tensor_b_shape_1,
    int64_t tensor_b_shape_2,
    const int32_t* indexes,
    int64_t indexes_shape_1
);


/// CUDA version of mops::outer_product_scatter_add for 32-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_f32(
    float* output,
    int64_t output_shape_1,
    int64_t output_shape_2,
    const float* tensor_a,
    int64_t tensor_a_shape_1,
    int64_t tensor_a_shape_2,
    const float* tensor_b,
    int64_t tensor_b_shape_1,
    int64_t tensor_b_shape_2,
    const int32_t* indexes,
    int64_t indexes_shape_1
);


/// CUDA version of mops::outer_product_scatter_add for 64-bit floats
int MOPS_EXPORT mops_cuda_outer_product_scatter_add_f64(
    double* output,
    int64_t output_shape_1,
    int64_t output_shape_2,
    const double* tensor_a,
    int64_t tensor_a_shape_1,
    int64_t tensor_a_shape_2,
    const double* tensor_b,
    int64_t tensor_b_shape_1,
    int64_t tensor_b_shape_2,
    const int32_t* indexes,
    int64_t indexes_shape_1
);


#ifdef __cplusplus
}
#endif


#endif
