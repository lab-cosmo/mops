
#include "mops/checks.hpp"
#include "mops/cuda_first_occurences.hpp"
#include "mops/cuda_utils.cuh"
#include "mops/opsa.hpp"

using namespace mops::cuda;

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4
#define FULL_MASK 0xffffffff

template <typename scalar_t, const int32_t TA, const int32_t TB>
__global__ __launch_bounds__(WARP_SIZE *NWARPS_PER_BLOCK) void outer_product_scatter_add_kernel(
    const scalar_t *__restrict__ A, // [nedges, nfeatures_A] - edge angular features
    const scalar_t *__restrict__ B, // [nedges, nfeatures_B] - radial/edge features
    const int32_t nnodes,           // number of nodes we're summing into
    const int32_t nedges_total,     // number of edges -> batch size of A and B
    const int32_t nfeatures_A,      // number of features of A
    const int32_t nfeatures_B,      // number of features of B
    const int32_t *first_occurences,
    const int32_t *__restrict__ indices_output, // sorted list of indices to sum
                                                // into [nedges]
    scalar_t *__restrict__ output               // shape: [nnodes, nfeatures_A, nfeatures_B] ->
                                                // this ordering because contiguity of threadCol
) {

    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;
    const int32_t nThreadRow = blockDim.x / WARP_SIZE;

    /* registers to hold components of A, B and output - used to increase
     * arithmetic intensity.
     */
    scalar_t regA[TA] = {0.0};
    scalar_t regB[TB] = {0.0};
    scalar_t regOP[TA * TB] = {0.0};

    const int32_t edge_start = first_occurences[blockIdx.x];
    const int32_t edge_end =
        (blockIdx.x == nnodes - 1) ? nedges_total : first_occurences[blockIdx.x + 1];
    const int32_t node_index = indices_output[edge_start];
    const int32_t nedges = edge_end - edge_start;

    /* total number of columns of A we can process is TA * WARP_SIZE, so
     * we need to loop find_integer_divisor(nfeatures_A, TA*WARP_SIZE) times
     */

    int32_t niter_A = find_integer_divisor(nfeatures_A, TA * nThreadRow);
    int32_t niter_B = find_integer_divisor(nfeatures_B, TB * WARP_SIZE);

    for (int32_t iter_B = 0; iter_B < niter_B; iter_B++) {
        int32_t global_B = iter_B * TB * WARP_SIZE;

        for (int32_t iter_A = 0; iter_A < niter_A; iter_A++) {
            int32_t global_A = iter_A * TA * nThreadRow;

            /*
             *  clear registers
             */
            for (int32_t i = 0; i < TA; i++) {
                regA[i] = 0;
            }

            for (int32_t i = 0; i < TB; i++) {
                regB[i] = 0;
            }

            for (int32_t i = 0; i < TA * TB; i++) {
                regOP[i] = 0.0;
            }

            for (int32_t edge_idx = 0; edge_idx < nedges; edge_idx++) {

                int32_t edge = edge_idx + edge_start;

                /*
                 *  load A from GMEM into local registers
                 */
                for (int32_t i = 0; i < TA; i++) {
                    if (global_A + i * nThreadRow + threadRow < nfeatures_A) {
                        regA[i] = A[edge * nfeatures_A + global_A + i * nThreadRow + threadRow];
                    }
                }

                /*
                 *  load B from GMEM into local registers
                 */
                for (int32_t i = 0; i < TB; i++) {
                    if (global_B + i * WARP_SIZE + threadCol < nfeatures_B) {
                        regB[i] = B[edge * nfeatures_B + global_B + i * WARP_SIZE + threadCol];
                    }
                }

                /*
                 * perform outer product in registers
                 */
                for (int32_t i = 0; i < TA; i++) {
                    for (int32_t j = 0; j < TB; j++) {
                        regOP[i * TB + j] += regA[i] * regB[j];
                    }
                }
            }

            /*
             * writeout the content of regOP to the output for this block of
             * [node, nfeatures_A, nfeatures_B]
             */
            for (int32_t j = 0; j < TB; j++) {
                if (global_B + j * WARP_SIZE + threadCol < nfeatures_B) {
                    for (int32_t i = 0; i < TA; i++) {
                        if (global_A + i * nThreadRow + threadRow < nfeatures_A) {
                            output
                                [node_index * nfeatures_B * nfeatures_A +
                                 (global_A + i * nThreadRow + threadRow) * nfeatures_B + global_B +
                                 j * WARP_SIZE + threadCol] = regOP[i * TB + j];
                        }
                    }
                }
            }
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

    int32_t nedges = A.shape[0];
    int32_t nnodes = output.shape[0];
    int32_t nfeatures_A = output.shape[1];
    int32_t nfeatures_B = output.shape[2];

    int32_t *first_occurences = calculate_first_occurences_cuda(indices_output.data, nedges, nnodes);

    dim3 gridDim(nnodes, 1, 1);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    outer_product_scatter_add_kernel<scalar_t, 2, 2><<<gridDim, blockDim, 0>>>(
        A.data,
        B.data,
        nnodes,
        nedges,
        nfeatures_A,
        nfeatures_B,
        first_occurences,
        indices_output.data,
        output.data
    );

    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B, Tensor<int32_t, 1> indices_output
);

template void mops::cuda::outer_product_scatter_add<double>(
    Tensor<double, 3> output, Tensor<double, 2> A, Tensor<double, 2> B, Tensor<int32_t, 1> indices_output
);

template <typename scalar_t>
__global__ void __launch_bounds__(NWARPS_PER_BLOCK *WARP_SIZE) outer_product_scatter_add_vjp_kernel(
    const scalar_t *__restrict__ A,               // [nedges, nfeatures_A] -> angular features
    const scalar_t *__restrict__ B,               // [nedges, nfeatures_B] -> radial +
                                                  // element features
    const int32_t nnodes,                         // number of nodes we're summing into
    const int32_t nedges_total,                   // number of edges -> batch size of A and B
    const int32_t nfeatures_A,                    // number of features of A
    const int32_t nfeatures_B,                    // number of features of B
    const int32_t *__restrict__ first_occurences, // indices in indices_output where
                                                  // the values change [nnodes]
    const int32_t *__restrict__ indices_output,   // sorted list of indices to
                                                  // sum into [nedges]
    scalar_t *__restrict__ grad_in,               // [nnodes, nfeatures_A, nfeatures_B]
    scalar_t *__restrict__ grad_A,                // [nedges, nfeatures_A]
    scalar_t *__restrict__ grad_B                 // [nedges, nfeatures_B]
) {

    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;
    const int32_t nThreadRow = blockDim.x / WARP_SIZE;

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_grad_in = shared_array<scalar_t>(nfeatures_A * nfeatures_B, sptr, &space);

    scalar_t *buffer_A = shared_array<scalar_t>(nThreadRow * nfeatures_A, sptr, &space);
    scalar_t *buffer_B = shared_array<scalar_t>(nThreadRow * nfeatures_B, sptr, &space);

    scalar_t *buffer_grad_A;
    scalar_t *buffer_grad_B;

    if (grad_A != nullptr) {
        buffer_grad_A = shared_array<scalar_t>(nThreadRow * nfeatures_A, sptr, &space);
    }

    if (grad_B != nullptr) {
        buffer_grad_B = shared_array<scalar_t>(nThreadRow * nfeatures_B, sptr, &space);
    }

    const int32_t edge_start = first_occurences[blockIdx.x];
    const int32_t edge_end =
        (blockIdx.x == nnodes - 1) ? nedges_total : first_occurences[blockIdx.x + 1];
    const int32_t node_index = indices_output[edge_start];
    const int32_t nedges = edge_end - edge_start;


    /*
     * initialise buffer_grad_in for this sub block
     */

    for (int tid = threadIdx.x; tid < nfeatures_A * nfeatures_B; tid += blockDim.x) {
        buffer_grad_in[tid] = grad_in[node_index * nfeatures_A * nfeatures_B + tid];
    }

    __syncthreads();

    for (int32_t edge_idx = threadRow; edge_idx < nedges; edge_idx += nThreadRow) {

        __syncwarp();

        int32_t edge = edge_idx + edge_start;

        /*
         * zero temporary buffers and load A, B into shared memory
         */

        for (int tid = threadCol; tid < nfeatures_A; tid += WARP_SIZE) {
            if (grad_A != nullptr) {
                buffer_grad_A[threadRow * nfeatures_A + tid] = 0.0;
            }
            buffer_A[threadRow * nfeatures_A + tid] = A[edge * nfeatures_A + tid];
        }

        for (int tid = threadCol; tid < nfeatures_B; tid += WARP_SIZE) {
            if (grad_B != nullptr) {
                buffer_grad_B[threadRow * nfeatures_B + tid] = 0.0;
            }
            buffer_B[threadRow * nfeatures_B + tid] = B[edge * nfeatures_B + tid];
        }

        __syncwarp();

        /*
         * perform the reduction
         */
        for (int i = 0; i < nfeatures_A; i++) {

            scalar_t dsumA = 0.0;

            for (int j = threadCol; j < nfeatures_B; j += WARP_SIZE) {

                scalar_t grad_in_ij = buffer_grad_in[i * nfeatures_B + j];

                if (grad_B != nullptr) {
                    buffer_grad_B[threadRow * nfeatures_B + j] +=
                        grad_in_ij * buffer_A[threadRow * nfeatures_A + i];
                }

                if (grad_A != nullptr) {
                    dsumA += grad_in_ij * buffer_B[threadRow * nfeatures_B + j];
                }
            }

            // need to warp shuffle reduce across the threads
            // accessing each B index.
            if (grad_A != nullptr) {
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

        if (grad_B != nullptr) {
            // write gradB
            for (int j = threadCol; j < nfeatures_B; j += WARP_SIZE) {
                grad_B[edge * nfeatures_B + j] = buffer_grad_B[threadRow * nfeatures_B + j];
            }
        }

        if (grad_A != nullptr) {
            // write gradA
            for (int i = threadCol; i < nfeatures_A; i += WARP_SIZE) {
                grad_A[edge * nfeatures_A + i] = buffer_grad_A[i * nThreadRow + threadRow];
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

    int32_t nedges = A.shape[0];
    int32_t nnodes = grad_output.shape[0];
    int32_t nfeatures_A = grad_output.shape[1];
    int32_t nfeatures_B = grad_output.shape[2];

    int32_t *first_occurences = calculate_first_occurences_cuda(indices_output.data, nedges, nnodes);

    dim3 gridDim(nnodes, 1, 1);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    void *sptr = 0;
    size_t space = 0;

    shared_array<scalar_t>(nfeatures_A * nfeatures_B, sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * nfeatures_A, sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * nfeatures_B, sptr, &space);

    if (grad_A.data != nullptr) {
        check_same_shape(grad_A, "grad_A", A, "A", "cuda_opsa_vjp");
        shared_array<scalar_t>(NWARPS_PER_BLOCK * nfeatures_A, sptr, &space);
    }
    if (grad_B.data != nullptr) {
        check_same_shape(grad_B, "grad_B", B, "B", "cuda_opsa_vjp");
        shared_array<scalar_t>(NWARPS_PER_BLOCK * nfeatures_B, sptr, &space);
    }

    outer_product_scatter_add_vjp_kernel<scalar_t><<<gridDim, blockDim, space>>>(
        A.data,
        B.data,
        nnodes,
        nedges,
        nfeatures_A,
        nfeatures_B,
        first_occurences,
        indices_output.data,
        grad_output.data,
        grad_A.data,
        grad_B.data
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
