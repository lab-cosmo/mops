#include "mops/cuda_first_occurences.hpp"
#include "mops/cuda_utils.cuh"

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

using namespace std;
using namespace mops::cuda;

__global__ void calculate_first_occurences_kernel(
    const int32_t *__restrict__ receiver_list, const int32_t nedges,
    const int32_t *__restrict__ sort_idx, bool use_sort,
    int32_t *__restrict__ first_occurences) {
    extern __shared__ char buffer[];
    size_t offset = 0;
    int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

    int32_t block_start = blockIdx.x * NEIGHBOUR_NEDGES_PER_BLOCK;

    // load all elements of senderlist needed by block into shared memory
    for (int32_t i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1;
         i += blockDim.x) {
        int32_t idx = block_start + i;

        if (idx < nedges) {
            if (use_sort) {
                smem[i] = receiver_list[sort_idx[idx]];
            } else {
                smem[i] = receiver_list[idx];
            }
        }
    }

    __syncthreads();

    // deal with even boundaries
    for (int32_t i = 2 * threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK;
         i += 2 * blockDim.x) {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges) {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2) {
                first_occurences[loc2] = idx + 1;
            }
        }
    }

    // deal with odd boundaries
    for (int32_t i = 2 * threadIdx.x + 1; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1;
         i += 2 * blockDim.x) {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges) {
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

/* from a sorted receiver list, computes the indices at which the reciever list
changes. I.e, gives the indiex range on the edges which should be summed into
the same reciever index.

if first_occurences is nullptr on entry, it will allocate the memory required
and return the alloc'd pointer. */

int32_t *mops::cuda::calculate_first_occurences_cuda(
    int32_t *receiver_list, int32_t nedges, int32_t natoms,
    int32_t *first_occurences = nullptr) {

    if (first_occurences == nullptr) {
        // cudamalloc it, return pointer reference later.
        CUDA_CHECK_ERROR(
            cudaMalloc(&first_occurences, natoms * sizeof(int32_t)));
    }

    int32_t nbx = find_integer_divisor(nedges, NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(64, 1, 1);

    size_t total_buff_size = (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

    calculate_first_occurences_kernel<<<block_dim, grid_dim, total_buff_size>>>(
        receiver_list, nedges, nullptr, false, first_occurences);

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    return first_occurences;
}