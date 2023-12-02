#include "cpu.tpp"

// explicit instanciations of templates
template void mops::sparse_accumulation_of_products<float>(
    Tensor<float, 2> output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<float, 1> C, Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B, Tensor<int32_t, 1> indices_output);

template void mops::sparse_accumulation_of_products<double>(
    Tensor<double, 2> output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<double, 1> C, Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B, Tensor<int32_t, 1> indices_output);

template void mops::sparse_accumulation_of_products_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::sparse_accumulation_of_products_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);


#ifdef MOPS_CUDA_ENABLED
#include "cuda.tpp"
#else
template <typename scalar_t>
void mops::cuda::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C, Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B, Tensor<int32_t, 1> indices_output) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::sparse_accumulation_of_products<float>(
    Tensor<float, 2> output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<float, 1> C, Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B, Tensor<int32_t, 1> indices_output);

template void mops::cuda::sparse_accumulation_of_products<double>(
    Tensor<double, 2> output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<double, 1> C, Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B, Tensor<int32_t, 1> indices_output);
