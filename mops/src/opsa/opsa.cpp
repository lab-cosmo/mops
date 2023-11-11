#include "cpu.tpp"

// explicit instanciations of templates
template void mops::outer_product_scatter_add<float>(
    Tensor<float, 2> output, Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_b, Tensor<int32_t, 1> indexes);

template void mops::outer_product_scatter_add<double>(
    Tensor<double, 2> output, Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_b, Tensor<int32_t, 1> indexes);

#ifdef MOPS_CUDA_ENABLED
#include "cuda.tpp"
#else
template <typename scalar_t>
void mops::cuda::outer_product_scatter_add(
    [[maybe_unused]] Tensor<scalar_t, 2> output,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_a,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_b,
    [[maybe_unused]] Tensor<int32_t, 1> indexes) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    Tensor<float, 2> output, Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_b, Tensor<int32_t, 1> indexes);

template void mops::cuda::outer_product_scatter_add<double>(
    Tensor<double, 2> output, Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_b, Tensor<int32_t, 1> indexes);
