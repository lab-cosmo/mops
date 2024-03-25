#include <cstdint>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_first_occurences.cuh"
#include "cuda_utils.cuh"

#define NELEMENTS_PER_BLOCK 512

using namespace std;

__global__ void calculate_first_occurences_kernel(
    const int32_t *__restrict__ receiver_list,
    const int32_t nelements,
    const int32_t *__restrict__ sort_idx,
    bool use_sort,
    int32_t *__restrict__ first_occurences
) {
    extern __shared__ char buffer[];
    size_t offset = 0;
    int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

    int32_t block_start = blockIdx.x * NELEMENTS_PER_BLOCK;

    // load all elements of senderlist needed by block into shared memory
    for (int32_t i = threadIdx.x; i < NELEMENTS_PER_BLOCK + 1; i += blockDim.x) {
        int32_t idx = block_start + i;

        if (idx < nelements) {
            if (use_sort) {
                smem[i] = receiver_list[sort_idx[idx]];
            } else {
                smem[i] = receiver_list[idx];
            }
        }
    }

    __syncthreads();

    // deal with even boundaries
    for (int32_t i = 2 * threadIdx.x; i < NELEMENTS_PER_BLOCK; i += 2 * blockDim.x) {
        int32_t idx = block_start + i;

        if (idx + 1 < nelements) {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2) {
                first_occurences[loc2] = idx + 1;
            }
        }
    }

    // deal with odd boundaries
    for (int32_t i = 2 * threadIdx.x + 1; i < NELEMENTS_PER_BLOCK + 1; i += 2 * blockDim.x) {
        int32_t idx = block_start + i;

        if (idx + 1 < nelements) {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2) {
                first_occurences[loc2] = idx + 1;
            }
        }
    }

    // deal with 0th element specifically, so we dont need to use torch::zeros
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        first_occurences[0] = 0;
    }
}

int32_t *calculate_first_occurences_cuda(
    const int32_t *receiver_list, int32_t nelements_input, int32_t nelements_output
) {

    static void *cached_first_occurences = nullptr;
    static size_t cached_size = 0;

    if (cached_size < nelements_output) {
        cudaMalloc(&cached_first_occurences, nelements_output * sizeof(int32_t));
        CUDA_CHECK_ERROR(cudaGetLastError());
        cached_size = nelements_output;
    }

    int32_t *result = reinterpret_cast<int32_t *>(cached_first_occurences);

    int32_t nbx = find_integer_divisor(nelements_input, NELEMENTS_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(64, 1, 1);

    size_t total_buff_size = (NELEMENTS_PER_BLOCK + 1) * sizeof(int32_t);

    calculate_first_occurences_kernel<<<block_dim, grid_dim, total_buff_size>>>(
        receiver_list, nelements_input, nullptr, false, result
    );

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    return result;
}
