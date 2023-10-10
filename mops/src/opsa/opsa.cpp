#include "cpu.tpp"

// explicit instanciations of templates
template void mops::outer_product_scatter_add<float>(
    float *output,
    size_t output_shape_1,
    size_t output_shape_2,
    const float *tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const float *tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t *indexes,
    size_t indexes_shape_1
);

template void mops::outer_product_scatter_add<double>(
    double *output,
    size_t output_shape_1,
    size_t output_shape_2,
    const double *tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const double *tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t *indexes,
    size_t indexes_shape_1
);


#ifdef MOPS_CUDA_ENABLED
    #include "cuda.tpp"
#else
template<typename scalar_t>
void mops::cuda::outer_product_scatter_add(
    scalar_t* output,
    size_t output_shape_1,
    size_t output_shape_2,
    const scalar_t* tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const scalar_t* tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t* indexes,
    size_t indexes_shape_1
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    float *output,
    size_t output_shape_1,
    size_t output_shape_2,
    const float *tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const float *tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t *indexes,
    size_t indexes_shape_1
);

template void mops::cuda::outer_product_scatter_add<double>(
    double *output,
    size_t output_shape_1,
    size_t output_shape_2,
    const double *tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const double *tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t *indexes,
    size_t indexes_shape_1
);
