#include "mops/sasaw.hpp"

#include "internal/checks.hpp"
#include "internal/cuda_first_occurences.cuh"
#include "internal/cuda_utils.cuh"

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

using namespace mops;
using namespace mops::cuda;

template <typename scalar_t>
__global__ __launch_bounds__(WARP_SIZE* NWARPS_PER_BLOCK) void sparse_accumulation_scatter_add_with_weights_kernel(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2,
    Tensor<int32_t, 1> first_occurences
) {

    extern __shared__ char buffer[];

    void* sptr = buffer;
    size_t space = 0;

    scalar_t* buffer_out = shared_array<scalar_t>(output.shape[1] * output.shape[2], sptr, &space);
    scalar_t* buffer_A = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    scalar_t* buffer_B = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * *W.shape[1] * W.shape[2], sptr, &space);

    scalar_t* buffer_C = shared_array<scalar_t>(C.shape[0], sptr, &space);
    int8_t* buffer_indices_A = shared_array<int8_t>(indices_A.shape[0], sptr, &space);
    int8_t* buffer_indices_W_2 = shared_array<int8_t>(indices_W_2.shape[0], sptr, &space);
    int8_t* buffer_indices_output_2 = shared_array<int8_t>(indices_output_2.shape[0], sptr, &space);

    int laneID = threadIdx.x % WARP_SIZE;
    int warpID = threadIdx.x / WARP_SIZE;

    int32_t sample_start = first_occurences.data[blockIdx.x];
    int32_t sample_end = -1;
    int32_t node_index = -1;

    if (sample_start != -1) {
        node_index = indices_output_1.data[sample_start];
        sample_end = (blockIdx.x == first_occurences.shape[0] - 1)
                         ? indices_output_1.shape[0]
                         : (first_occurences.data[blockIdx.x + 1] == -1
                                ? indices_output_1.shape[0]
                                : first_occurences.data[blockIdx.x + 1]);
    }

    int32_t nsamples = sample_end - sample_start;

    if (nsamples == 0) {
        return;
    }

    for (int tid = threadIdx.x; tid < indices_A.shape[0]; tid += blockDim.x) {
        buffer_indices_A[tid] = (int8_t)indices_A.data[tid];
        buffer_indices_W_2[tid] = (int8_t)indices_W_2.data[tid];
        buffer_C[tid] = C.data[tid];
        buffer_indices_output_2[tid] = (int8_t)indices_output_2.data[tid];
    }

    for (int tid = threadIdx.x; tid < output.shape[1] * output.shape[2]; tid += blockDim.x) {
        buffer_out[tid] = 0.0;
    }

    __syncthreads();

    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {

        int sample = sample_start + sample_idx + warpID;

        if (sample >= sample_end) {
            break;
        }

        // load in temporary buffers for each sample

        for (int tid = laneID; tid < A.shape[1]; tid += WARP_SIZE) {
            buffer_A[warpID * A.shape[1] + tid] = A.data[sample * A.shape[1] + tid];
        }

        for (int tid = laneID; tid < B.shape[1]; tid += WARP_SIZE) {
            buffer_B[warpID * B.shape[1] + tid] = B.data[sample * B.shape[1] + tid];
        }

        for (int j = 0; j < W.shape[1]; j++) {
            for (int tid = laneID; tid < W.shape[2]; tid += WARP_SIZE) {
                buffer_W[warpID * W.shape[1] * W.shape[2] + j * W.shape[2] + tid] =
                    W.data[indices_W_1.data[sample] * W.shape[1] * W.shape[2] + j * W.shape[2] + tid];
            }
        }

        __syncwarp();

        for (int k = 0; k < C.shape[0]; k++) {
            for (int tid = laneID; tid < W.shape[2]; tid += WARP_SIZE) {
                scalar_t w =
                    buffer_W[warpID * W.shape[1] * W.shape[2] + buffer_indices_W_2[k] * W.shape[2] + tid];

                int32_t a_idx = buffer_indices_A[k];
                int32_t index_output_2 = buffer_indices_output_2[k];

                atomicAdd(
                    &buffer_out[index_output_2 * output.shape[2] + tid],
                    A.data[sample * A.shape[1] + a_idx] * B[sample * B.shape[1] + tid] *
                        buffer_C[k] * w
                );
            }
        }
    }

    __syncthreads();

    for (int i = warpID; i < output.shape[1]; i += NWARPS_PER_BLOCK) {
        for (int j = laneID; j < output.shape[2]; j += WARP_SIZE) {
            output.data[node_index * output.shape[1] * output.shape[2] + i * output.shape[2] + j] =
                buffer_out[i * output.shape[2] + j];
        }
    }
}

template <typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
) {
    // TODO
}

// explicit instantiations of CUDA templates
template void mops::cuda::sparse_accumulation_scatter_add_with_weights<float>(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights<double>(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

template <typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_W,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<scalar_t, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
) {
    // TODO
}

template void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2
);
