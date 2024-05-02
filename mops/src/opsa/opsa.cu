#include "mops/opsa.hpp"

#include "internal/checks/opsa.hpp"
#include "internal/cuda_first_occurences.cuh"
#include "internal/cuda_utils.cuh"

using namespace mops;
using namespace mops::cuda;

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4
#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void outer_product_scatter_add_kernel(
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> first_occurences,
    Tensor<int32_t, 1> indices_output,
    Tensor<scalar_t, 3> output
) {

    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;

    int32_t* first_occurences_start = first_occurences.data;
    int32_t* first_occurences_end = first_occurences.data + output.shape[0];

    int32_t sample_start = first_occurences_start[blockIdx.x];
    int32_t sample_end = first_occurences_end[blockIdx.x];

    int32_t nsamples = sample_end - sample_start;

    if (nsamples == 0) {
        // fill tensor with zeros instead
        for (int i = threadRow; i < A.shape[1]; i += NWARPS_PER_BLOCK) {
            for (int j = threadCol; j < B.shape[1]; j += WARP_SIZE) {
                output.data[blockIdx.x * A.shape[1] * B.shape[1] + i * B.shape[1] + j] = 0.0;
            }
        }
        return;
    }

    for (int i = threadRow; i < A.shape[1]; i += NWARPS_PER_BLOCK) {
        for (int j = threadCol; j < B.shape[1]; j += WARP_SIZE) {

            scalar_t reg_output = 0.0;

            for (int32_t sample_idx = 0; sample_idx < nsamples; sample_idx++) {

                int32_t sample = sample_idx + sample_start;

                reg_output += A.data[sample * A.shape[1] + i] * B.data[sample * B.shape[1] + j];
            }

            output.data[blockIdx.x * A.shape[1] * B.shape[1] + i * B.shape[1] + j] = reg_output;
        }
    }
}

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
) {
    check_opsa(output, A, B, indices_output, "cuda_outer_product_scatter_add");

    cudaPointerAttributes attributes;
    CUDA_CHECK_ERROR(cudaPointerGetAttributes(&attributes, A.data));
    int current_device;
    CUDA_CHECK_ERROR(cudaGetDevice(&current_device));
    if (current_device != attributes.device) {
        CUDA_CHECK_ERROR(cudaSetDevice(attributes.device));
    }

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    int32_t* first_occurences = calculate_first_occurences_cuda(
        indices_output.data, indices_output.shape[0], output.shape[0]
    );

    dim3 gridDim(output.shape[0], 1, 1);

    dim3 blockDim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

    outer_product_scatter_add_kernel<scalar_t><<<gridDim, blockDim, 0, cstream>>>(
        A, B, mops::Tensor<int32_t, 1>{first_occurences, {output.shape[0] * 2}}, indices_output, output
    );

    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(cstream));

    if (current_device != attributes.device) {
        CUDA_CHECK_ERROR(cudaSetDevice(current_device));
    }
}

// explicit instantiations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

template void mops::cuda::outer_product_scatter_add<double>(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

template <typename scalar_t>
__global__ void outer_product_scatter_add_vjp_kernel(
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

    void* sptr = buffer;
    size_t space = 0;

    scalar_t* buffer_grad_in = shared_array<scalar_t>(A.shape[1] * B.shape[1], sptr, &space);

    scalar_t* buffer_A = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    scalar_t* buffer_B = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);

    scalar_t* buffer_grad_A;
    scalar_t* buffer_grad_B;

    if (grad_A.data != nullptr) {
        buffer_grad_A = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    }

    if (grad_B.data != nullptr) {
        buffer_grad_B = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    }

    int32_t* first_occurences_start = first_occurences.data;
    int32_t* first_occurences_end = first_occurences.data + grad_in.shape[0];

    int32_t sample_start = first_occurences_start[blockIdx.x];
    int32_t sample_end = first_occurences_end[blockIdx.x];

    int32_t nsamples = sample_end - sample_start;

    if (nsamples == 0) {
        return;
    }

    /*
     * initialise buffer_grad_in for this sub block
     */

    for (int tid = threadIdx.x; tid < A.shape[1] * B.shape[1]; tid += blockDim.x) {
        buffer_grad_in[tid] = grad_in.data[blockIdx.x * A.shape[1] * B.shape[1] + tid];
    }

    __syncthreads();

    for (int32_t sample_idx = threadRow; sample_idx < nsamples; sample_idx += NWARPS_PER_BLOCK) {

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
                    buffer_grad_A[i * NWARPS_PER_BLOCK + threadRow] = dsumA;
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
                grad_A.data[sample * A.shape[1] + i] =
                    buffer_grad_A[i * NWARPS_PER_BLOCK + threadRow];
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
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
) {
    check_opsa_vjp(
        grad_A, grad_B, grad_output, A, B, indices_output, "cuda_outer_product_scatter_add_vjp"
    );

    cudaPointerAttributes attributes;
    CUDA_CHECK_ERROR(cudaPointerGetAttributes(&attributes, A.data));
    int current_device;
    CUDA_CHECK_ERROR(cudaGetDevice(&current_device));
    if (current_device != attributes.device) {
        CUDA_CHECK_ERROR(cudaSetDevice(attributes.device));
    }

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

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
        shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    }
    if (grad_B.data != nullptr) {
        shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    }

    outer_product_scatter_add_vjp_kernel<scalar_t><<<gridDim, blockDim, space, cstream>>>(
        A,
        B,
        mops::Tensor<int32_t, 1>{first_occurences, {grad_output.shape[0]}},
        indices_output,
        grad_output,
        grad_A,
        grad_B
    );

    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(cstream));

    if (current_device != attributes.device) {
        CUDA_CHECK_ERROR(cudaSetDevice(current_device));
    }
}

// these templates will be precompiled and provided in the mops library
template void mops::cuda::outer_product_scatter_add_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

template void mops::cuda::outer_product_scatter_add_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);

template <typename scalar_t>
__global__ void outer_product_scatter_add_vjp_vjp_kernel(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> first_occurences,
    Tensor<int32_t, 1> indices_output
) {
    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;

    void* sptr = buffer;
    size_t space = 0;

    scalar_t* buffer_grad_out;

    scalar_t* buffer_A;
    scalar_t* buffer_B;

    scalar_t* buffer_grad_grad_out;
    scalar_t* buffer_grad_A_2;
    scalar_t* buffer_grad_B_2;

    scalar_t* buffer_grad_grad_A;
    scalar_t* buffer_grad_grad_B;

    // if ((grad_grad_A.data != nullptr && grad_B_2.data != nullptr) ||
    //     (grad_grad_B.data != nullptr && grad_A_2.data != nullptr)) {
    buffer_grad_out = shared_array<scalar_t>(A.shape[1] * B.shape[1], sptr, &space);
    //}

    if (grad_grad_output.data != nullptr &&
        (grad_grad_A.data != nullptr || grad_grad_B.data != nullptr)) {
        buffer_grad_grad_out = shared_array<scalar_t>(A.shape[1] * B.shape[1], sptr, &space);
        buffer_A = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
        buffer_B = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    }

    if (grad_A_2.data != nullptr) {
        buffer_grad_A_2 = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    }

    if (grad_B_2.data != nullptr) {
        buffer_grad_B_2 = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    }

    if (grad_grad_output.data != nullptr) {
        if (grad_B_2.data != nullptr && grad_grad_A.data != nullptr) {
            // initialise grad_grad_A buffer
            buffer_grad_grad_A = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
        }

        if (grad_A_2.data != nullptr && grad_grad_B.data != nullptr) {
            // initialise grad_grad_B buffer
            buffer_grad_grad_B = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
        }
    }

    int32_t* first_occurences_start = first_occurences.data;
    int32_t* first_occurences_end = first_occurences.data + grad_output.shape[0];

    int32_t sample_start = first_occurences_start[blockIdx.x];
    int32_t sample_end = first_occurences_end[blockIdx.x];

    int32_t nsamples = sample_end - sample_start;

    if (nsamples == 0) {
        return;
    }

    /*
     * initialise buffer_grad_in for this sub block
     */

    if ((grad_grad_A.data != nullptr && grad_B_2.data != nullptr) ||
        (grad_grad_B.data != nullptr && grad_A_2.data != nullptr)) {
        for (int tid = threadIdx.x; tid < A.shape[1] * B.shape[1]; tid += blockDim.x) {
            buffer_grad_out[tid] = grad_output.data[blockIdx.x * A.shape[1] * B.shape[1] + tid];
        }
    }

    if (grad_grad_output.data != nullptr &&
        (grad_grad_A.data != nullptr || grad_grad_B.data != nullptr)) {
        for (int tid = threadIdx.x; tid < A.shape[1] * B.shape[1]; tid += blockDim.x) {
            buffer_grad_grad_out[tid] = 0.0;
        }
    }

    __syncthreads();

    for (int32_t sample_idx = threadRow; sample_idx < nsamples; sample_idx += NWARPS_PER_BLOCK) {

        __syncwarp();

        int32_t sample = sample_idx + sample_start;

        /*
         * zero temporary buffers and load A, B into shared memory
         */

        for (int tid = threadCol; tid < A.shape[1]; tid += WARP_SIZE) {

            if (grad_A_2.data != nullptr) {
                buffer_grad_A_2[threadRow * A.shape[1] + tid] = 0.0;
            }
            if (grad_grad_output.data != nullptr && grad_grad_B.data != nullptr) {
                buffer_A[threadRow * A.shape[1] + tid] = A.data[sample * A.shape[1] + tid];
            }

            if (grad_grad_output.data != nullptr && grad_B_2.data != nullptr &&
                grad_grad_A.data != nullptr) {
                buffer_grad_grad_A[threadRow * A.shape[1] + tid] =
                    grad_grad_A.data[sample * A.shape[1] + tid];
            }
        }

        for (int tid = threadCol; tid < B.shape[1]; tid += WARP_SIZE) {
            if (grad_B_2.data != nullptr) {
                buffer_grad_B_2[threadRow * B.shape[1] + tid] = 0.0;
            }

            if (grad_grad_output.data != nullptr && grad_grad_A.data != nullptr) {
                buffer_B[threadRow * B.shape[1] + tid] = B.data[sample * B.shape[1] + tid];
            }
            if (grad_grad_output.data != nullptr && grad_A_2.data != nullptr &&
                grad_grad_B.data != nullptr) {
                // initialise grad_grad_B buffer
                buffer_grad_grad_B[threadRow * B.shape[1] + tid] =
                    grad_grad_B.data[sample * B.shape[1] + tid];
            }
        }

        __syncwarp();

        /*
         * perform the reduction
         */
        for (int i = 0; i < A.shape[1]; i++) {

            scalar_t grad_A2_tmp = 0.0;

            for (int j = threadCol; j < B.shape[1]; j += WARP_SIZE) {

                if (grad_A_2.data != nullptr && grad_grad_B.data != nullptr) {
                    grad_A2_tmp += buffer_grad_grad_B[threadRow * B.shape[1] + j] *
                                   buffer_grad_out[i * B.shape[1] + j];
                }

                if (grad_B_2.data != nullptr && grad_grad_A.data != nullptr) {
                    buffer_grad_B_2[threadRow * B.shape[1] + j] +=
                        buffer_grad_grad_A[threadRow * A.shape[1] + i] *
                        buffer_grad_out[i * B.shape[1] + j];
                }

                if (grad_grad_output.data != nullptr) {

                    if (grad_grad_B.data != nullptr) {
                        buffer_grad_grad_out[i * B.shape[1] + j] +=
                            buffer_A[threadRow * A.shape[1] + i] *
                            buffer_grad_grad_B[threadRow * B.shape[1] + j];
                    }

                    if (grad_grad_A.data != nullptr) {
                        buffer_grad_grad_out[i * B.shape[1] + j] +=
                            buffer_B[threadRow * B.shape[1] + j] *
                            buffer_grad_grad_A[threadRow * A.shape[1] + i];
                    }
                }
            }

            // reduce across B dimension
            if (grad_A_2.data != nullptr && grad_grad_B.data != nullptr) {
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    grad_A2_tmp += __shfl_down_sync(FULL_MASK, grad_A2_tmp, offset, WARP_SIZE);
                }

                if (threadCol == 0) {
                    buffer_grad_A_2[i * NWARPS_PER_BLOCK + threadRow] = grad_A2_tmp;
                }
            }
        }

        __syncwarp();

        if (grad_B_2.data != nullptr && grad_grad_A.data != nullptr) {
            // write gradB
            for (int j = threadCol; j < B.shape[1]; j += WARP_SIZE) {
                grad_B_2.data[sample * B.shape[1] + j] = buffer_grad_B_2[threadRow * B.shape[1] + j];
            }
        }

        if (grad_A_2.data != nullptr && grad_grad_B.data != nullptr) {
            // write gradA
            for (int i = threadCol; i < A.shape[1]; i += WARP_SIZE) {
                grad_A_2.data[sample * A.shape[1] + i] =
                    buffer_grad_A_2[i * NWARPS_PER_BLOCK + threadRow];
            }
        }
    }

    if (grad_grad_output.data != nullptr &&
        (grad_grad_A.data != nullptr || grad_grad_B.data != nullptr)) {
        for (int tid = threadIdx.x; tid < A.shape[1] * B.shape[1]; tid += blockDim.x) {
            grad_grad_output.data[blockIdx.x * A.shape[1] * B.shape[1] + tid] =
                buffer_grad_grad_out[tid];
        }
    }
}

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_vjp_vjp(
    Tensor<scalar_t, 3> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
) {

    check_opsa_vjp_vjp(
        grad_grad_output,
        grad_A_2,
        grad_B_2,
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        indices_output,
        "cuda_outer_product_scatter_add_vjp_vjp"
    );

    cudaPointerAttributes attributes;
    CUDA_CHECK_ERROR(cudaPointerGetAttributes(&attributes, A.data));
    int current_device;
    CUDA_CHECK_ERROR(cudaGetDevice(&current_device));
    if (current_device != attributes.device) {
        CUDA_CHECK_ERROR(cudaSetDevice(attributes.device));
    }

    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    int32_t* first_occurences = calculate_first_occurences_cuda(
        indices_output.data, indices_output.shape[0], grad_output.shape[0]
    );

    dim3 gridDim(grad_output.shape[0], 1, 1);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    void* sptr = 0;
    size_t space = 0;

    scalar_t* buffer_grad_out;

    scalar_t* buffer_A;
    scalar_t* buffer_B;

    scalar_t* buffer_grad_grad_out;
    scalar_t* buffer_grad_A_2;
    scalar_t* buffer_grad_B_2;

    scalar_t* buffer_grad_grad_A;
    scalar_t* buffer_grad_grad_B;

    buffer_grad_out = shared_array<scalar_t>(A.shape[1] * B.shape[1], sptr, &space);

    if (grad_grad_output.data != nullptr &&
        (grad_grad_A.data != nullptr || grad_grad_B.data != nullptr)) {
        buffer_grad_grad_out = shared_array<scalar_t>(A.shape[1] * B.shape[1], sptr, &space);
        buffer_A = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
        buffer_B = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    }

    if (grad_A_2.data != nullptr) {
        buffer_grad_A_2 = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
    }

    if (grad_B_2.data != nullptr) {
        buffer_grad_B_2 = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
    }

    if (grad_grad_output.data != nullptr) {
        if (grad_B_2.data != nullptr && grad_grad_A.data != nullptr) {
            // initialise grad_grad_A buffer
            buffer_grad_grad_A = shared_array<scalar_t>(NWARPS_PER_BLOCK * A.shape[1], sptr, &space);
        }

        if (grad_A_2.data != nullptr && grad_grad_B.data != nullptr) {
            // initialise grad_grad_B buffer
            buffer_grad_grad_B = shared_array<scalar_t>(NWARPS_PER_BLOCK * B.shape[1], sptr, &space);
        }
    }

    outer_product_scatter_add_vjp_vjp_kernel<scalar_t><<<gridDim, blockDim, space, cstream>>>(
        grad_grad_output,
        grad_A_2,
        grad_B_2,
        grad_grad_A,
        grad_grad_B,
        grad_output,
        A,
        B,
        mops::Tensor<int32_t, 1>{first_occurences, {grad_output.shape[0]}},
        indices_output
    );

    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(cstream));

    if (current_device != attributes.device) {
        CUDA_CHECK_ERROR(cudaSetDevice(current_device));
    }
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
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
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
    Tensor<int32_t, 1> indices_output,
    void* cuda_stream
);
