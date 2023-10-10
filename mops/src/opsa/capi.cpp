#include "mops/capi.hpp"

#include "mops/opsa.hpp"
#include "mops/opsa.h"


extern "C" int mops_outer_product_scatter_add_f32(
    float* output,
    size_t output_shape_1,
    size_t output_shape_2,
    const float* tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const float* tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t* indexes,
    size_t indexes_shape_1
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add(
            output,
            output_shape_1,
            output_shape_2,
            tensor_a,
            tensor_a_shape_1,
            tensor_a_shape_2,
            tensor_b,
            tensor_b_shape_1,
            tensor_b_shape_2,
            indexes,
            indexes_shape_1
        );
    );
}


extern "C" int mops_outer_product_scatter_add_f64(
    double* output,
    size_t output_shape_1,
    size_t output_shape_2,
    const double* tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const double* tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t* indexes,
    size_t indexes_shape_1
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add(
            output,
            output_shape_1,
            output_shape_2,
            tensor_a,
            tensor_a_shape_1,
            tensor_a_shape_2,
            tensor_b,
            tensor_b_shape_1,
            tensor_b_shape_2,
            indexes,
            indexes_shape_1
        );
    );
}


extern "C" int mops_cuda_outer_product_scatter_add_f32(
    float* output,
    size_t output_shape_1,
    size_t output_shape_2,
    const float* tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const float* tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t* indexes,
    size_t indexes_shape_1
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add(
            output,
            output_shape_1,
            output_shape_2,
            tensor_a,
            tensor_a_shape_1,
            tensor_a_shape_2,
            tensor_b,
            tensor_b_shape_1,
            tensor_b_shape_2,
            indexes,
            indexes_shape_1
        );
    );
}


extern "C" int mops_cuda_outer_product_scatter_add_f64(
    double* output,
    size_t output_shape_1,
    size_t output_shape_2,
    const double* tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const double* tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t* indexes,
    size_t indexes_shape_1
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add(
            output,
            output_shape_1,
            output_shape_2,
            tensor_a,
            tensor_a_shape_1,
            tensor_a_shape_2,
            tensor_b,
            tensor_b_shape_1,
            tensor_b_shape_2,
            indexes,
            indexes_shape_1
        );
    );
}
