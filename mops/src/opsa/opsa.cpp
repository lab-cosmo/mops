#include "cpu.tpp"

// explicit instanciations of templates
template void
mops::outer_product_scatter_add<float>(Tensor<float, 2> output,
                                       Tensor<float, 2> A, Tensor<float, 2> B,
                                       Tensor<int32_t, 1> indices_output);

template void mops::outer_product_scatter_add<double>(
    Tensor<double, 2> output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output);

template void mops::outer_product_scatter_add_vjp<float>(
    Tensor<float, 2> grad_A, Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output);

template void mops::outer_product_scatter_add_vjp<double>(
    Tensor<double, 2> grad_A, Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output);

#ifdef MOPS_CUDA_ENABLED
#include "cuda.tpp"
#else
template <typename scalar_t>
void mops::cuda::outer_product_scatter_add(Tensor<scalar_t, 2>,
                                           Tensor<scalar_t, 2>,
                                           Tensor<scalar_t, 2>,
                                           Tensor<int32_t, 1>) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    Tensor<float, 2> output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output);

template void mops::cuda::outer_product_scatter_add<double>(
    Tensor<double, 2> output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output);
