#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#include "mops/checks.hpp"
#include "mops/cuda_utils.cuh"
#include "mops/hpe.hpp"

using namespace mops;
using namespace mops::cuda;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

using namespace mops;
using namespace mops::cuda;

template <typename scalar_t, int32_t polynomial_order>
__global__ void homogeneous_polynomial_evaluation_kernel(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C, Tensor<int32_t, 2> indices_A
) {
    extern __shared__ char buffer[];

    int32_t nbatch = A.shape[0];
    int32_t nnu1 = A.shape[1];
    int32_t nbasis = C.shape[0];

    void* sptr = buffer;
    size_t space = 0;

    /* shared buffers */
    scalar_t* buffer_nu1 = shared_array<scalar_t>(nnu1, sptr, &space);
    scalar_t* tmp_sum = shared_array<scalar_t>(blockDim.x / WARP_SIZE, sptr, &space);

    int32_t batch_id = blockIdx.x;

    if (batch_id > nbatch) {
        return;
    }

    // load all of A into shared memory
    for (int32_t i = threadIdx.x; i < nnu1; i += blockDim.x) {
        buffer_nu1[i] = A.data[batch_id * nnu1 + i];
    }

    __syncthreads();

    scalar_t batch_sum = 0.0;
    scalar_t c = 0.0; // kahans summation

    if (threadIdx.x == 0) {
        output.data[batch_id] = 0.0;
    }

    for (int32_t basis = threadIdx.x; basis < nbasis; basis += blockDim.x) {

        int16_t idx = indices_A.data[0 * indices_A.shape[0] + basis];
        scalar_t tmp = buffer_nu1[idx];

#pragma unroll
        for (int32_t i_monomial = 1; i_monomial < polynomial_order; i_monomial++) {
            idx = indices_A.data[i_monomial * indices_A.shape[0] + basis];

            tmp *= buffer_nu1[idx];
        }

        // kahans summation for error control
        scalar_t y = tmp * C.data[basis] - c;
        scalar_t t = batch_sum + y;
        c = (t - batch_sum) - y;
        batch_sum = t;
    }

    for (int32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        batch_sum += __shfl_down_sync(FULL_MASK, batch_sum, offset);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        tmp_sum[threadIdx.x / WARP_SIZE] = batch_sum;
    }

    __syncthreads();

    scalar_t out = 0.0;

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            out += tmp_sum[i];
        }

        output.data[batch_id] = out;
    }
}

template <typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C, Tensor<int32_t, 2> indices_A
) {

    int nbatch = output.shape[0];
    int nnu1 = A.shape[1];

    dim3 block_dim(nbatch);

    dim3 thread_block(128, 1, 1);

    void* sptr = nullptr;
    size_t space = 0;

    shared_array<scalar_t>(nnu1, sptr, &space);
    shared_array<scalar_t>(thread_block.x / WARP_SIZE, sptr, &space);

    size_t polynomial_order = indices_A.shape[1];

    if (polynomial_order <= 10) {
        switch (polynomial_order) {
        case 0:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 0>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 1:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 1>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 2:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 2>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 3:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 3>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 4:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 4>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 5:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 5>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 6:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 6>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 7:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 7>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 8:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 8>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 9:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 9>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        case 10:
            homogeneous_polynomial_evaluation_kernel<scalar_t, 10>
                <<<block_dim, thread_block, space>>>(output, A, C, indices_A);
            break;
        default:
            break;
        }
    }

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// explicit instanciations of CUDA templates
template void mops::cuda::homogeneous_polynomial_evaluation<float>(
    Tensor<float, 1> output, Tensor<float, 2> A, Tensor<float, 1> C, Tensor<int32_t, 2> indices_A
);

template void mops::cuda::homogeneous_polynomial_evaluation<double>(
    Tensor<double, 1> output, Tensor<double, 2> A, Tensor<double, 1> C, Tensor<int32_t, 2> indices_A
);