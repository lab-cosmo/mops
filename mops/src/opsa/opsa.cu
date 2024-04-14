#include "mops/opsa.hpp"

#include "internal/checks.hpp"
#include "internal/cuda_first_occurences.cuh"
#include "internal/cuda_utils.cuh"

using namespace mops;
using namespace mops::cuda;

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4
#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ __launch_bounds__(WARP_SIZE* NWARPS_PER_BLOCK) void outer_product_scatter_add_kernel(
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> first_occurences,
    Tensor<int32_t, 1> indices_output,
    Tensor<scalar_t, 3> output
) {

    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;
    const int32_t nThreadRow = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    int32_t sample_start = first_occurences.data[blockIdx.x];
    int32_t sample_end = -1;
    int32_t node_index = -1;

    if (sample_start != -1) {
        node_index = indices_output.data[sample_start];
        sample_end = (blockIdx.x == first_occurences.shape[0] - 1)
                         ? indices_output.shape[0]
                         : (first_occurences.data[blockIdx.x + 1] == -1
                                ? indices_output.shape[0]
                                : first_occurences.data[blockIdx.x + 1]);
    }

    int32_t nsamples = sample_end - sample_start;

    if (nsamples == 0) {
        return;
    }

    for (int i = threadRow; i < A.shape[1]; i += nThreadRow) {
        for (int j = threadCol; j < B.shape[1]; j += WARP_SIZE) {

            scalar_t reg_output = 0.0;

            for (int32_t sample_idx = 0; sample_idx < nsamples; sample_idx++) {

                int32_t sample = sample_idx + sample_start;

                reg_output += A.data[sample * A.shape[1] + i] * B.data[sample * B.shape[1] + j];
            }

            output.data[node_index * A.shape[1] * B.shape[1] + i * B.shape[1] + j] = reg_output;
        }
    }
}

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {

    check_sizes(A, "A", 0, B, "B", 0, "opsa");
    check_sizes(A, "A", 1, output, "output", 1, "opsa");
    check_sizes(B, "B", 1, output, "output", 2, "opsa");
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, "opsa");

    int32_t* first_occurences = calculate_first_occurences_cuda(
        indices_output.data, indices_output.shape[0], output.shape[0]
    );

    dim3 gridDim(output.shape[0], 1, 1);

    dim3 blockDim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

    outer_product_scatter_add_kernel<scalar_t><<<gridDim, blockDim, 0>>>(
        A, B, mops::Tensor<int32_t, 1>{first_occurences, {output.shape[0]}}, indices_output, output
    );

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// explicit instantiations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B, Tensor<int32_t, 1> indices_output
);

template void mops::cuda::outer_product_scatter_add<double>(
    Tensor<double, 3> output, Tensor<double, 2> A, Tensor<double, 2> B, Tensor<int32_t, 1> indices_output
);

template <typename scalar_t>
__global__ void __launch_bounds__(NWARPS_PER_BLOCK* WARP_SIZE) outer_product_scatter_add_vjp_kernel(
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> first_occurences,
    Tensor<int32_t, 1> indices_output,
    Tensor<scalar_t, 3> grad_in,
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B
) {

    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;
    const int32_t nThreadRow = blockDim.x / WARP_SIZE;

    void* sptr = buffer;
    size_t space = 0;

    scalar_t* buffer_grad_in = shared_array<scalar_t>(A.shape[1] * B.shape[1], sptr, &space);

    scalar_t* buffer_A = shared_array<scalar_t>(nThreadRow * A.shape[1], sptr, &space);
    scalar_t* buffer_B = shared_array<scalar_t>(nThreadRow * B.shape[1], sptr, &space);

    scalar_t* buffer_grad_A;
    scalar_t* buffer_grad_B;

    if (grad_A.data != nullptr) {
        buffer_grad_A = shared_array<scalar_t>(nThreadRow * A.shape[1], sptr, &space);
    }

    if (grad_B.data != nullptr) {
        buffer_grad_B = shared_array<scalar_t>(nThreadRow * B.shape[1], sptr, &space);
    }

    int32_t sample_start = first_occurences.data[blockIdx.x];
    int32_t sample_end = -1;
    int32_t node_index = -1;

    if (sample_start != -1) {
        node_index = indices_output.data[sample_start];
        sample_end = (blockIdx.x == first_occurences.shape[0] - 1)
                         ? indices_output.shape[0]
                         : (first_occurences.data[blockIdx.x + 1] == -1
                                ? indices_output.shape[0]
                                : first_occurences.data[blockIdx.x + 1]);
    }

    int32_t nsamples = sample_end - sample_start;

    if (nsamples == 0) {
        return;
    }

    /*
     * initialise buffer_grad_in for this sub block
     */

    for (int tid = threadIdx.x; tid < A.shape[1] * B.shape[1]; tid += blockDim.x) {
        buffer_grad_in[tid] = grad_in.data[node_index * A.shape[1] * B.shape[1] + tid];
    }

    __syncthreads();

    for (int32_t sample_idx = threadRow; sample_idx < nsamples; sample_idx += nThreadRow) {

        __syncwarp();

        int32_t sample = sample_idx + sample_start;

        /*
         * zero temporary buffers and load A, B into shared memory
         */

        for (int tid = threadCol; tid < A.shape[1]; tid += WARP_SIZE) {
            if (grad_A.data != nullptr) {
                buffer_grad_A[threadRow * A.shape[1] + tid] = 0.0;
            }
            buffer_A[threadRow * A.shape[1] + tid] = A.data[sample * A.shape[1] + tid];
        }

        for (int tid = threadCol; tid < B.shape[1]; tid += WARP_SIZE) {
            if (grad_B.data != nullptr) {
                buffer_grad_B[threadRow * B.shape[1] + tid] = 0.0;
            }
            buffer_B[threadRow * B.shape[1] + tid] = B.data[sample * B.shape[1] + tid];
        }

        __syncwarp();

        /*
         * perform the reduction
         */
        for (int i = 0; i < A.shape[1]; i++) {

            scalar_t dsumA = 0.0;

            for (int j = threadCol; j < B.shape[1]; j += WARP_SIZE) {

                scalar_t grad_in_ij = buffer_grad_in[i * B.shape[1] + j];

                if (grad_B.data != nullptr) {
                    buffer_grad_B[threadRow * B.shape[1] + j] +=
                        grad_in_ij * buffer_A[threadRow * A.shape[1] + i];
                }

                if (grad_A.data != nullptr) {
                    dsumA += grad_in_ij * buffer_B[threadRow * B.shape[1] + j];
                }
            }

            // need to warp shuffle reduce across the threads
            // accessing each B index.
            if (grad_A.data != nullptr) {
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    dsumA += __shfl_down_sync(FULL_MASK, dsumA, offset, WARP_SIZE);
                }

                // thread 0 contains the gradient for this subset of features_A.
                if (threadCol == 0) {
                    buffer_grad_A[i * nThreadRow + threadRow] = dsumA;
                }
            }
        }

        __syncwarp();

        if (grad_B.data != nullptr) {
            // write gradB
            for (int j = threadCol; j < B.shape[1]; j += WARP_SIZE) {
                grad_B.data[sample * B.shape[1] + j] = buffer_grad_B[threadRow * B.shape[1] + j];
            }
        }

        if (grad_A.data != nullptr) {
            // write gradA
            for (int i = threadCol; i < A.shape[1]; i += WARP_SIZE) {
                grad_A.data[sample * A.shape[1] + i] = buffer_grad_A[i * nThreadRow + threadRow];
            }
        }
    }
}

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {

    check_sizes(A, "A", 0, B, "B", 0, "cuda_opsa_vjp");
    check_sizes(A, "A", 1, grad_output, "grad_output", 1, "cuda_opsa_vjp");
    check_sizes(B, "B", 1, grad_output, "grad_output", 2, "cuda_opsa_vjp");
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, "cuda_opsa_vjp");

    int32_t* first_occurences = calculate_first_occurences_cuda(
        indices_output.data, indices_output.shape[0], grad_output.shape[0]
    );

    dim3 gridDim(grad_output.shape[0], 1, 1);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    void* sptr = 0;
    size_t space = 0;

    shared_array<scalar_t>(A.shape[1] * B.shape[1], sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);

    if (grad_A.data != nullptr) {
        check_same_shape(grad_A, "grad_A", A, "A", "cuda_opsa_vjp");
        shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    }
    if (grad_B.data != nullptr) {
        check_same_shape(grad_B, "grad_B", B, "B", "cuda_opsa_vjp");
        shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    }

    outer_product_scatter_add_vjp_kernel<scalar_t><<<gridDim, blockDim, space>>>(
        A,
        B,
        mops::Tensor<int32_t, 1>{first_occurences, {grad_output.shape[0]}},
        indices_output,
        grad_output,
        grad_A,
        grad_B
    );

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// these templates will be precompiled and provided in the mops library
template void mops::cuda::outer_product_scatter_add_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::outer_product_scatter_add_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_vjp_vjp(
    Tensor<scalar_t, 3> /*grad_grad_output*/,
    Tensor<scalar_t, 2> /*grad_A_2*/,
    Tensor<scalar_t, 2> /*grad_B_2*/,
    Tensor<scalar_t, 2> /*grad_grad_A*/,
    Tensor<scalar_t, 2> /*grad_grad_B*/,
    Tensor<scalar_t, 3> /*grad_output*/,
    Tensor<scalar_t, 2> /*A*/,
    Tensor<scalar_t, 2> /*B*/,
    Tensor<int32_t, 1> /*indices_output*/
) {
    throw std::runtime_error("Not implemented");
}

// explicit instantiations of CUDA templates
template void mops::cuda::outer_product_scatter_add_vjp_vjp<float>(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::outer_product_scatter_add_vjp_vjp<double>(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);
