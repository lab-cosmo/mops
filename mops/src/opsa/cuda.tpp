#include <stdexcept>

#include "mops/cuda_opsa.hpp"
#include "mops/opsa.hpp"

using namespace mops::cuda;

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add(Tensor<scalar_t, 2> output,
                                           Tensor<scalar_t, 2> A,
                                           Tensor<scalar_t, 2> B,
                                           Tensor<int32_t, 1> indices_output) {

    // invoke the kernel launch wrapper.
    outer_product_scatter_add_cuda<scalar_t>(
        A.data,              // [nedges, nfeatures_A]
        B.data,              // [nedges, nfeatures_B]
        output.shape[0],     // nnodes
        A.shape[0],          // nedges
        A.shape[1],          // nfeatures_A
        B.shape[1],          // nfeatures_B
        indices_output.data, // [nedges]
        output.data          // [nnodes, nfeatures_A, nfeatures_B]
    );
}

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    Tensor<float, 2> output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output);

template void mops::cuda::outer_product_scatter_add<double>(
    Tensor<double, 2> output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output);

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A, Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output, Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B, Tensor<int32_t, 1> indices_output) {

    outer_product_scatter_add_vjp_cuda<scalar_t>(
        A.data,               // [nedges, nfeatures_A]
        B.data,               // [nedges, nfeatures_B]
        grad_output.shape[0], // // nnodes
        A.shape[0],           // nedges
        A.shape[1],           // nfeatures_A
        B.shape[1],           // nfeatures_B
        indices_output.data,  // [nedges]
        grad_output.data,     //[nnodes, nfeatures_B * nfeatures_A]
        grad_A.data,          //[nnodes, nfeatures_A]
        grad_B.data           //  //[nnodes, nfeatures_B]
    );
}

// these templates will be precompiled and provided in the mops library
template void mops::cuda::outer_product_scatter_add_vjp<float>(
    Tensor<float, 2> grad_A, Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output);

template void mops::cuda::outer_product_scatter_add_vjp<double>(
    Tensor<double, 2> grad_A, Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output);
