#include "mops/sap.hpp"

#include "internal/checks.hpp"
#include "internal/cuda_utils.cuh"

using namespace mops;
using namespace mops::cuda;

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void sparse_accumulation_of_products_kernel(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    extern __shared__ char buffer[];

    void* sptr = buffer;
    size_t space = 0;

    /* shared buffers */
    scalar_t* buffer_out = shared_array<scalar_t>(WARP_SIZE * output.shape[1], sptr, &space);
    scalar_t* buffer_A = shared_array<scalar_t>(WARP_SIZE * A.shape[1], sptr, &space);
    scalar_t* buffer_B = shared_array<scalar_t>(WARP_SIZE * B.shape[1], sptr, &space);

    int32_t* packed_indices = shared_array<int32_t>(indices_A.shape[0], sptr, &space);

    int32_t laneID = threadIdx.x % WARP_SIZE;
    int32_t rowID = threadIdx.x / WARP_SIZE;
    int32_t nRows = find_integer_divisor(blockDim.x, WARP_SIZE);

    for (int32_t i = threadIdx.x; i < indices_A.shape[0]; i += blockDim.x) {
        packed_indices[i] =
            indices_A.data[i] << 16 | indices_B.data[i] << 8 | indices_output.data[i];
    }

    int32_t idx_start = blockIdx.x * WARP_SIZE;

    for (int i = rowID; i < A.shape[1]; i += nRows) {

        if ((idx_start + laneID) * A.shape[1] + i < A.shape[0] * A.shape[1]) {
            buffer_A[i * WARP_SIZE + laneID] = A.data[(idx_start + laneID) * A.shape[1] + i];
        }
    }
    /*
    for (int i = threadIdx.x; i < WARP_SIZE * A.shape[1]; i += blockDim.x) {
        int col = i % A.shape[1];
        int row = i / A.shape[1];

        buffer_A[col * WARP_SIZE + row] = A.data[(idx_start + row) * A.shape[1] + col];
    }*/

    /*for (int i = threadIdx.x; i < WARP_SIZE * B.shape[1]; i += blockDim.x) {
        int col = i % B.shape[1];
        int row = i / B.shape[1];

        buffer_B[col * WARP_SIZE + row] = B.data[(idx_start + row) * B.shape[1] + col];
    }*/

    for (int i = rowID; i < B.shape[1]; i += nRows) {

        if ((idx_start + laneID) * B.shape[1] + i < B.shape[0] * B.shape[1]) {
            buffer_B[i * WARP_SIZE + laneID] = B.data[(idx_start + laneID) * B.shape[1] + i];
        }
    }

    for (int idx = threadIdx.x; idx < WARP_SIZE * output.shape[1]; idx += blockDim.x) {
        buffer_out[idx] = 0.0;
    }

    __syncthreads();

    for (int k = rowID; k < C.shape[0]; k += nRows) {

        int out_idx = packed_indices[k] & 0xFF;
        int b_idx = (packed_indices[k] >> 8) & 0xFF;
        int a_idx = (packed_indices[k] >> 16) & 0xFF;

        atomicAdd(
            buffer_out + out_idx * WARP_SIZE + laneID,
            C.data[k] * buffer_A[a_idx * WARP_SIZE + laneID] * buffer_B[b_idx * WARP_SIZE + laneID]
        );
    }

    __syncthreads();

    for (int i = threadIdx.x; i < WARP_SIZE * output.shape[1]; i += blockDim.x) {
        int col = i % output.shape[1];
        int row = i / output.shape[1];

        output.data[(idx_start + row) * output.shape[1] + col] = buffer_out[col * WARP_SIZE + row];
    }
}

template <typename scalar_t>
void mops::cuda::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {

    dim3 block_dim(find_integer_divisor(A.shape[0], WARP_SIZE));

    dim3 thread_block(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

    void* sptr = nullptr;
    size_t space = 0;

    shared_array<scalar_t>(WARP_SIZE * output.shape[1], sptr, &space);
    shared_array<scalar_t>(WARP_SIZE * A.shape[1], sptr, &space);
    shared_array<scalar_t>(WARP_SIZE * B.shape[1], sptr, &space);

    sparse_accumulation_of_products_kernel<scalar_t>
        <<<block_dim, thread_block, space>>>(output, A, B, C, indices_A, indices_B, indices_output);

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// explicit instanciations of CUDA templates
template void mops::cuda::sparse_accumulation_of_products<float>(
    Tensor<float, 2> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::sparse_accumulation_of_products<double>(
    Tensor<double, 2> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template <typename scalar_t>
void mops::cuda::sparse_accumulation_of_products_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    // TODO
}

template void mops::cuda::sparse_accumulation_of_products_vjp<float>(
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

template void mops::cuda::sparse_accumulation_of_products_vjp<double>(
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