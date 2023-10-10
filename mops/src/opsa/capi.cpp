#include "mops/capi.hpp"

#include "mops/opsa.hpp"
#include "mops/opsa.h"

static size_t checked_cast(int64_t value) {
    if (value < 0 || static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error(
            "integer value '" + std::to_string(value) + "' does not fit in size_t"
        );
    }
    return static_cast<size_t>(value);
}


extern "C" int mops_outer_product_scatter_add_f32(
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
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add(
            output,
            checked_cast(output_shape_1),
            checked_cast(output_shape_2),
            tensor_a,
            checked_cast(tensor_a_shape_1),
            checked_cast(tensor_a_shape_2),
            tensor_b,
            checked_cast(tensor_b_shape_1),
            checked_cast(tensor_b_shape_2),
            indexes,
            checked_cast(indexes_shape_1)
        );
    );
}


extern "C" int mops_outer_product_scatter_add_f64(
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
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::outer_product_scatter_add(
            output,
            checked_cast(output_shape_1),
            checked_cast(output_shape_2),
            tensor_a,
            checked_cast(tensor_a_shape_1),
            checked_cast(tensor_a_shape_2),
            tensor_b,
            checked_cast(tensor_b_shape_1),
            checked_cast(tensor_b_shape_2),
            indexes,
            checked_cast(indexes_shape_1)
        );
    );
}


extern "C" int mops_cuda_outer_product_scatter_add_f32(
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
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add(
            output,
            checked_cast(output_shape_1),
            checked_cast(output_shape_2),
            tensor_a,
            checked_cast(tensor_a_shape_1),
            checked_cast(tensor_a_shape_2),
            tensor_b,
            checked_cast(tensor_b_shape_1),
            checked_cast(tensor_b_shape_2),
            indexes,
            checked_cast(indexes_shape_1)
        );
    );
}


extern "C" int mops_cuda_outer_product_scatter_add_f64(
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
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::outer_product_scatter_add(
            output,
            checked_cast(output_shape_1),
            checked_cast(output_shape_2),
            tensor_a,
            checked_cast(tensor_a_shape_1),
            checked_cast(tensor_a_shape_2),
            tensor_b,
            checked_cast(tensor_b_shape_1),
            checked_cast(tensor_b_shape_2),
            indexes,
            checked_cast(indexes_shape_1)
        );
    );
}
