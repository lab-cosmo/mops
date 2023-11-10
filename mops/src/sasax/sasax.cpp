#include "cpu.tpp"

// explicit instanciations of templates
template void mops::sparse_accumulation_scatter_add_with_weights<float>(
    Tensor<float, 3> output, Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_r, Tensor<float, 3> tensor_x,
    Tensor<float, 1> tensor_c, Tensor<int, 1> tensor_i, Tensor<int, 1> tensor_j,
    Tensor<int, 1> tensor_m_1, Tensor<int, 1> tensor_m_2,
    Tensor<int, 1> tensor_m_3);

template void mops::sparse_accumulation_scatter_add_with_weights<double>(
    Tensor<double, 3> output, Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_r, Tensor<double, 3> tensor_x,
    Tensor<double, 1> tensor_c, Tensor<int, 1> tensor_i,
    Tensor<int, 1> tensor_j, Tensor<int, 1> tensor_m_1,
    Tensor<int, 1> tensor_m_2, Tensor<int, 1> tensor_m_3);

#ifdef MOPS_CUDA_ENABLED
#include "cuda.tpp"
#else
template <typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output, Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_r, Tensor<scalar_t, 3> tensor_x,
    Tensor<scalar_t, 1> tensor_c, Tensor<int, 1> tensor_i,
    Tensor<int, 1> tensor_j, Tensor<int, 1> tensor_m_1,
    Tensor<int, 1> tensor_m_2, Tensor<int, 1> tensor_m_3) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::sparse_accumulation_scatter_add_with_weights<float>(
    Tensor<float, 3> output, Tensor<float, 2> tensor_a,
    Tensor<float, 2> tensor_r, Tensor<float, 3> tensor_x,
    Tensor<float, 1> tensor_c, Tensor<int, 1> tensor_i, Tensor<int, 1> tensor_j,
    Tensor<int, 1> tensor_m_1, Tensor<int, 1> tensor_m_2,
    Tensor<int, 1> tensor_m_3);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights<double>(
    Tensor<double, 3> output, Tensor<double, 2> tensor_a,
    Tensor<double, 2> tensor_r, Tensor<double, 3> tensor_x,
    Tensor<double, 1> tensor_c, Tensor<int, 1> tensor_i,
    Tensor<int, 1> tensor_j, Tensor<int, 1> tensor_m_1,
    Tensor<int, 1> tensor_m_2, Tensor<int, 1> tensor_m_3);
