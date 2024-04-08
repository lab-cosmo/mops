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

    for (int i = rowID; i < output.shape[1]; i += nRows) {

        if ((idx_start + laneID) * output.shape[1] + i < output.shape[0] * output.shape[1]) {
            output.data[(idx_start + laneID) * output.shape[1] + i] =
                buffer_out[i * WARP_SIZE + laneID];
        }
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
    shared_array<int32_t>(indices_A.shape[0], sptr, &space);

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
__global__ void sparse_accumulation_of_products_vjp_kernel(
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
    extern __shared__ char buffer[];

    void* sptr = buffer;
    size_t space = 0;

    /* shared buffers */
    scalar_t* buffer_gradout =
        shared_array<scalar_t>(WARP_SIZE * grad_output.shape[1], sptr, &space);
    int32_t* packed_indices = shared_array<int32_t>(indices_A.shape[0], sptr, &space);

    scalar_t* buffer_A;
    scalar_t* buffer_B;
    scalar_t* buffer_gradA;
    scalar_t* buffer_gradB;

    if (grad_B.data != nullptr) {
        buffer_A = shared_array<scalar_t>(WARP_SIZE * A.shape[1], sptr, &space);
        buffer_gradB = shared_array<scalar_t>(WARP_SIZE * grad_B.shape[1], sptr, &space);
    }

    if (grad_A.data != nullptr) {
        buffer_B = shared_array<scalar_t>(WARP_SIZE * B.shape[1], sptr, &space);
        buffer_gradA = shared_array<scalar_t>(WARP_SIZE * grad_A.shape[1], sptr, &space);
    }

    int32_t laneID = threadIdx.x % WARP_SIZE;
    int32_t rowID = threadIdx.x / WARP_SIZE;
    int32_t nRows = find_integer_divisor(blockDim.x, WARP_SIZE);

    int32_t idx_start = blockIdx.x * WARP_SIZE;

    for (int32_t i = threadIdx.x; i < indices_A.shape[0]; i += blockDim.x) {
        packed_indices[i] =
            indices_A.data[i] << 16 | indices_B.data[i] << 8 | indices_output.data[i];
    }

    if (grad_A.data) {
        for (int i = threadIdx.x; i < WARP_SIZE * grad_A.shape[1]; i += blockDim.x) {
            buffer_gradA[i] = 0.0;
        }

        for (int i = rowID; i < B.shape[1]; i += nRows) {

            if ((idx_start + laneID) * B.shape[1] + i < B.shape[0] * B.shape[1]) {
                buffer_B[i * WARP_SIZE + laneID] = B.data[(idx_start + laneID) * B.shape[1] + i];
            }
        }
    }

    if (grad_B.data) {
        for (int i = threadIdx.x; i < WARP_SIZE * grad_B.shape[1]; i += blockDim.x) {
            buffer_gradB[i] = 0.0;
        }

        for (int i = rowID; i < A.shape[1]; i += nRows) {

            if ((idx_start + laneID) * A.shape[1] + i < A.shape[0] * A.shape[1]) {
                buffer_A[i * WARP_SIZE + laneID] = A.data[(idx_start + laneID) * A.shape[1] + i];
            }
        }
    }

    for (int i = rowID; i < grad_output.shape[1]; i += nRows) {

        if ((idx_start + laneID) * grad_output.shape[1] + i <
            grad_output.shape[0] * grad_output.shape[1]) {
            buffer_gradout[i * WARP_SIZE + laneID] =
                grad_output.data[(idx_start + laneID) * grad_output.shape[1] + i];
        }
    }

    __syncthreads();

    for (int k = rowID; k < C.shape[0]; k += nRows) {

        int out_idx = packed_indices[k] & 0xFF;
        int b_idx = (packed_indices[k] >> 8) & 0xFF;
        int a_idx = (packed_indices[k] >> 16) & 0xFF;

        if (grad_A.data != nullptr) {
            atomicAdd(
                buffer_gradA + a_idx * WARP_SIZE + laneID,
                C.data[k] * buffer_B[b_idx * WARP_SIZE + laneID] *
                    buffer_gradout[out_idx * WARP_SIZE + laneID]
            );
        }
        if (grad_B.data != nullptr) {
            atomicAdd(
                buffer_gradB + b_idx * WARP_SIZE + laneID,
                C.data[k] * buffer_A[a_idx * WARP_SIZE + laneID] *
                    buffer_gradout[out_idx * WARP_SIZE + laneID]
            );
        }
    }

    __syncthreads();
    if (grad_A.data != nullptr) {
        for (int i = rowID; i < grad_A.shape[1]; i += nRows) {

            if ((idx_start + laneID) * grad_A.shape[1] + i < grad_A.shape[0] * grad_A.shape[1]) {
                grad_A.data[(idx_start + laneID) * grad_A.shape[1] + i] =
                    buffer_gradA[i * WARP_SIZE + laneID];
            }
        }
    }

    if (grad_B.data != nullptr) {
        for (int i = rowID; i < grad_B.shape[1]; i += nRows) {

            if ((idx_start + laneID) * grad_B.shape[1] + i < grad_B.shape[0] * grad_B.shape[1]) {
                grad_B.data[(idx_start + laneID) * grad_B.shape[1] + i] =
                    buffer_gradB[i * WARP_SIZE + laneID];
            }
        }
    }
}
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
    dim3 block_dim(find_integer_divisor(grad_A.shape[0], WARP_SIZE));

    dim3 thread_block(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

    void* sptr = nullptr;
    size_t space = 0;

    shared_array<scalar_t>(WARP_SIZE * grad_output.shape[1], sptr, &space);
    shared_array<int32_t>(indices_A.shape[0], sptr, &space);
    
    if (grad_B.data != nullptr) {
        shared_array<scalar_t>(WARP_SIZE * A.shape[1], sptr, &space);
        shared_array<scalar_t>(WARP_SIZE * grad_B.shape[1], sptr, &space);
    }

    if (grad_A.data != nullptr) {
        shared_array<scalar_t>(WARP_SIZE * B.shape[1], sptr, &space);
        shared_array<scalar_t>(WARP_SIZE * grad_A.shape[1], sptr, &space);
    }

    sparse_accumulation_of_products_vjp_kernel<scalar_t><<<block_dim, thread_block, space>>>(
        grad_A, grad_B, grad_output, A, B, C, indices_A, indices_B, indices_output
    );

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
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