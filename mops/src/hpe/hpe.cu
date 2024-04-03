#include "mops/hpe.hpp"

#include "internal/checks.hpp"
#include "internal/cuda_utils.cuh"

using namespace mops;
using namespace mops::cuda;

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

#define FULL_MASK 0xffffffff

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
    scalar_t* tmp_sum = shared_array<scalar_t>(NWARPS_PER_BLOCK, sptr, &space);
    int32_t* buffer_indices_A = shared_array<int32_t>(blockDim.x * polynomial_order, sptr, &space);

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

    if (threadIdx.x == 0) {
        output.data[batch_id] = 0.0;
    }

    // indices_A : nbasis, polynomial_order
    for (int32_t i = 0; i < nbasis; i += blockDim.x) {

        __syncthreads();

        int32_t i_monomial = threadIdx.x % polynomial_order;
        int32_t x = threadIdx.x / polynomial_order;
        int32_t nx = blockDim.x / polynomial_order;

        for (int lbasis = x; lbasis < blockDim.x; lbasis += nx) {
            if (i_monomial * blockDim.x + lbasis < polynomial_order * blockDim.x) {
                buffer_indices_A[i_monomial * blockDim.x + lbasis] =
                    indices_A.data[(i + lbasis) * polynomial_order + i_monomial];
            }
        }

        __syncthreads();

        int32_t basis = i + threadIdx.x;

        if (basis < nbasis) {

            scalar_t tmp = 1.0;

            for (int32_t i_monomial = 0; i_monomial < polynomial_order; i_monomial++) {
                int32_t idx = buffer_indices_A[i_monomial * blockDim.x + threadIdx.x];

                tmp *= buffer_nu1[idx];
            }

            batch_sum += tmp * C.data[basis];
        }
    }

    __syncthreads();

    for (int32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        batch_sum += __shfl_down_sync(FULL_MASK, batch_sum, offset);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        tmp_sum[threadIdx.x / WARP_SIZE] = batch_sum;
    }

    __syncthreads();

    if (threadIdx.x == 0) {

        scalar_t out = 0.0;

        for (int32_t i = 0; i < NWARPS_PER_BLOCK; i++) {
            out += tmp_sum[i];
        }

        output.data[batch_id] = out;
    }
}

template <typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C, Tensor<int32_t, 2> indices_A
) {

    int32_t nbatch = output.shape[0];
    int32_t nnu1 = A.shape[1];
    size_t polynomial_order = indices_A.shape[1];

    dim3 block_dim(nbatch);

    dim3 thread_block(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

    void* sptr = nullptr;
    size_t space = 0;

    shared_array<scalar_t>(nnu1, sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK, sptr, &space);
    shared_array<int32_t>(WARP_SIZE * NWARPS_PER_BLOCK * polynomial_order, sptr, &space);

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

template <typename scalar_t, int32_t polynomial_order>
__global__ void homogeneous_polynomial_evaluation_vjp_kernel(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A

) {
    extern __shared__ char buffer[];

    int32_t nbatch = A.shape[0];
    int32_t nnu1 = A.shape[1];
    int32_t nbasis = C.shape[0];

    void* sptr = buffer;
    size_t space = 0;

    /* shared buffers */
    scalar_t* buffer_nu1 = shared_array<scalar_t>(nnu1, sptr, &space);
    scalar_t* buffer_gradA = shared_array<scalar_t>(nnu1, sptr, &space);
    int32_t* buffer_indices_A = shared_array<int32_t>(blockDim.x * polynomial_order, sptr, &space);

    int32_t batch_id = blockIdx.x;

    if (batch_id > nbatch) {
        return;
    }

    // load all of A into shared memory
    for (int32_t i = threadIdx.x; i < nnu1; i += blockDim.x) {
        buffer_nu1[i] = A.data[batch_id * nnu1 + i];
        buffer_gradA[i] = 0.0;
    }

    __syncthreads();

    scalar_t gout = grad_output.data[batch_id];

    // indices_A : nbasis, polynomial_order
    for (int32_t i = 0; i < nbasis; i += blockDim.x) {

        __syncthreads();

        int32_t i_monomial = threadIdx.x % polynomial_order;
        int32_t x = threadIdx.x / polynomial_order;
        int32_t nx = blockDim.x / polynomial_order;

        for (int lbasis = x; lbasis < blockDim.x; lbasis += nx) {
            if (i_monomial * blockDim.x + lbasis < polynomial_order * blockDim.x) {
                buffer_indices_A[i_monomial * blockDim.x + lbasis] =
                    indices_A.data[(i + lbasis) * polynomial_order + i_monomial];
            }
        }

        __syncthreads();

        int32_t basis = i + threadIdx.x;

        if (basis < nbasis) {

            scalar_t c = C.data[basis] * gout;

            for (int32_t i_monomial = 0; i_monomial < polynomial_order; i_monomial++) {

                scalar_t tmp_i = c;

                for (int32_t j_monomial = 0; j_monomial < polynomial_order; j_monomial++) {

                    if (i_monomial == j_monomial) {
                        continue;
                    }

                    int32_t idx_j =
                        buffer_indices_A[j_monomial * blockDim.x + threadIdx.x]; // indices_A.data[j_monomial
                                                                                 // * indices_A.shape[0] + basis];

                    tmp_i *= buffer_nu1[idx_j];
                }

                int32_t idx_i = buffer_indices_A[i_monomial * blockDim.x + threadIdx.x];

                atomicAdd(&buffer_gradA[idx_i], tmp_i);
            }
        }
    }

    __syncthreads();

    for (int32_t i = threadIdx.x; i < nnu1; i += blockDim.x) {
        grad_A.data[batch_id * nnu1 + i] = buffer_gradA[i];
    }
}

template <typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A
) {

    int32_t nbatch = grad_output.shape[0];
    int32_t nnu1 = A.shape[1];
    size_t polynomial_order = indices_A.shape[1];

    dim3 block_dim(nbatch);

    dim3 thread_block(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    void* sptr = nullptr;
    size_t space = 0;

    shared_array<scalar_t>(2 * nnu1, sptr, &space);
    shared_array<int32_t>(NWARPS_PER_BLOCK * WARP_SIZE * polynomial_order, sptr, &space);

    if (polynomial_order <= 10) {
        switch (polynomial_order) {
        case 0:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 0>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 1:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 1>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 2:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 2>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 3:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 3>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 4:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 4>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 5:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 5>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 6:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 6>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 7:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 7>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 8:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 8>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 9:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 9>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        case 10:
            homogeneous_polynomial_evaluation_vjp_kernel<scalar_t, 10>
                <<<block_dim, thread_block, space>>>(grad_A, grad_output, A, C, indices_A);
            break;
        default:
            break;
        }
    }

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// explicit instanciations of CUDA templates
template void mops::cuda::homogeneous_polynomial_evaluation_vjp<float>(
    Tensor<float, 2> gradA,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A
);

template void mops::cuda::homogeneous_polynomial_evaluation_vjp<double>(
    Tensor<double, 2> gradA,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);