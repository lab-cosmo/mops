#include "cpu.tpp"

// explicit instanciations of templates
template void mops::outer_product_scatter_add_with_weights<float>(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j
);

template void mops::outer_product_scatter_add_with_weights<double>(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j
);


#ifdef MOPS_CUDA_ENABLED
    #include "cuda.tpp"
#else
template<typename scalar_t>
void mops::cuda::outer_product_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 2> W,
    Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add_with_weights<float>(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 2> W,
    Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j
);

template void mops::cuda::outer_product_scatter_add_with_weights<double>(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 2> W,
    Tensor<int32_t, 1> i,
    Tensor<int32_t, 1> j
);
