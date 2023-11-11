#include "cpu.tpp"

// explicit instanciations of templates
template void mops::outer_product_scatter_add_with_weights<float>(
    Tensor<float, 3> output, Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_r, Tensor<float, 2> tensor_x,
    Tensor<int32_t, 1> tensor_i, Tensor<int32_t, 1> tensor_j);

template void mops::outer_product_scatter_add_with_weights<double>(
    Tensor<double, 3> output, Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_r, Tensor<double, 2> tensor_x,
    Tensor<int32_t, 1> tensor_i, Tensor<int32_t, 1> tensor_j);

#ifdef MOPS_CUDA_ENABLED
#include "cuda.tpp"
#else
template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_with_weights(
    [[maybe_unused]] Tensor<scalar_t, 3> output,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_a,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_r,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_x,
    [[maybe_unused]] Tensor<int32_t, 1> tensor_i,
    [[maybe_unused]] Tensor<int32_t, 1> tensor_j) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add_with_weights<float>(
    Tensor<float, 3> output, Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_r, Tensor<float, 2> tensor_x,
    Tensor<int32_t, 1> tensor_i, Tensor<int32_t, 1> tensor_j);

template void mops::cuda::outer_product_scatter_add_with_weights<double>(
    Tensor<double, 3> output, Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_r, Tensor<double, 2> tensor_x,
    Tensor<int32_t, 1> tensor_i, Tensor<int32_t, 1> tensor_j);
