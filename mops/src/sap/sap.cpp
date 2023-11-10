#include "cpu.tpp"

// explicit instanciations of templates
template void mops::sparse_accumulation_of_products<float>(
    Tensor<float, 2> output,
    Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_b,
    Tensor<int32_t, 1> indexes
);

template void mops::sparse_accumulation_of_products<double>(
    Tensor<double, 2> output,
    Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_b,
    Tensor<int32_t, 1> indexes
);


#ifdef MOPS_CUDA_ENABLED
    #include "cuda.tpp"
#else
template<typename scalar_t>
void mops::cuda::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_b,
    Tensor<int32_t, 1> indexes
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::sparse_accumulation_of_products<float>(
    Tensor<float, 2> output,
    Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_b,
    Tensor<int32_t, 1> indexes
);

template void mops::cuda::sparse_accumulation_of_products<double>(
    Tensor<double, 2> output,
    Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_b,
    Tensor<int32_t, 1> indexes
);
