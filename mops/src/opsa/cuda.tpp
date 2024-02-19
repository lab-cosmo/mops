#include <stdexcept>

#include "mops/opsa.hpp"
#include "mops/opsa_cuda.cuh"

using namespace mops::cuda;

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add(Tensor<scalar_t, 2> output,
                                           Tensor<scalar_t, 2> A,
                                           Tensor<scalar_t, 2> B,
                                           Tensor<int32_t, 1> first_occurences,
                                           Tensor<int32_t, 1> indices_output) {

    // invoke the kernel launch wrapper.
    outer_product_scatter_add_cuda<scalar_t>(A.data, // [nedges, nfeatures_B]
                                             B.data, // [nedges, nfeatures_B]
                                             output.shape[0], // nnodes
                                             A.shape[0],      // nedges
                                             A.shape[1],      // nfeatures_A
                                             B.shape[1],      // nfeatures_B
                                             first_occurences.data, //
                                             indices_output.data,   //
                                             output.data);
}

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A, Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output, Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B, Tensor<int32_t, 1> first_occurences,
    Tensor<int32_t, 1> indices_output) {}