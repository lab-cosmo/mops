
#include "mops/cuda_opsa.hpp"
#include "mops/cuda_utils.cuh"

using namespace mops::cuda;

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4
#define FULL_MASK 0xffffffff

template <typename scalar_t, const int32_t TA, const int32_t TB>
__global__
__launch_bounds__(WARP_SIZE *NWARPS_PER_BLOCK) void outer_product_scatter_add_kernel(
    const scalar_t *__restrict__ A, // [nedges, nfeatures_A]
    const scalar_t *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,           // number of nodes we're summing into
    const int32_t nedges_total,     // number of edges -> batch size of A and B
    const int32_t nfeatures_A,      // number of features of A
    const int32_t nfeatures_B,      // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where the
                                        // values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to sum
                                                // into [nedges]
    scalar_t
        *__restrict__ output // shape: [nnodes, nfeatures_B, nfeatures_A] ->
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
    const int32_t edge_end = (blockIdx.x == nnodes - 1)
                                 ? nedges_total
                                 : first_occurences[blockIdx.x + 1];
    const int32_t node_index = indices_output[edge_start];
    const int32_t nedges = edge_end - edge_start;

    /* total number of columns of A we can process is TA * WARP_SIZE, so
     * we need to loop find_integer_divisor(nfeatures_A, TA*WARP_SIZE) times
     */

    int32_t niter_A = find_integer_divisor(nfeatures_A, TA * WARP_SIZE);
    int32_t niter_B = find_integer_divisor(nfeatures_B, TB * nThreadRow);

    for (int32_t iter_B = 0; iter_B < niter_B; iter_B++) {
        int32_t global_B = iter_B * TB * nThreadRow;

        for (int32_t iter_A = 0; iter_A < niter_A; iter_A++) {
            int32_t global_A = iter_A * TA * WARP_SIZE;

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

                    if (global_A + i * WARP_SIZE + threadCol < nfeatures_A)
                        regA[i] = A[edge * nfeatures_A + global_A +
                                    i * WARP_SIZE + threadCol];
                }

                /*
                 *  load B from GMEM into local registers
                 */
                for (int32_t i = 0; i < TB; i++) {
                    if (global_B + i * nThreadRow + threadRow < nfeatures_B)
                        regB[i] = B[edge * nfeatures_B + global_B +
                                    i * nThreadRow + threadRow];
                }

                /*
                 * perform outer product in registers
                 */
                for (int32_t j = 0; j < TB; j++) {
                    for (int32_t i = 0; i < TA; i++) {
                        regOP[j * TA + i] += regA[i] * regB[j];
                    }
                }
            }

            /*
             * writeout the content of regOP to the output for this block of
             * [node, nfeatures_B, nfeatures_A]
             */
            for (int32_t j = 0; j < TB; j++) {
                if (global_B + j * nThreadRow + threadRow < nfeatures_B) {
                    for (int32_t i = 0; i < TA; i++) {
                        if (global_A + i * WARP_SIZE + threadCol <
                            nfeatures_A) {
                            output[node_index * nfeatures_B * nfeatures_A +
                                   (global_B + j * nThreadRow + threadRow) *
                                       nfeatures_A +
                                   global_A + i * WARP_SIZE + threadCol] =
                                regOP[j * TA + i];
                        }
                    }
                }
            }
        }
    }
}

template <typename scalar_t, const int TB>
__global__ void __launch_bounds__(NWARPS_PER_BLOCK *WARP_SIZE)
    outer_product_scatter_add_vjp_kernel(
        const scalar_t
            *__restrict__ A, // [nedges, nfeatures_A] -> angular features
        const scalar_t *__restrict__ B, // [nedges, nfeatures_B] -> radial +
                                        // element features
        const int32_t nnodes,           // number of nodes we're summing into
        const int32_t nedges_total, // number of edges -> batch size of A and B
        const int32_t nfeatures_A,  // number of features of A
        const int32_t nfeatures_B,  // number of features of B
        const int32_t
            *__restrict__ first_occurences, // indices in indices_output where
                                            // the values change [nnodes]
        const int32_t *__restrict__ indices_output, // sorted list of indices to
                                                    // sum into [nedges]
        scalar_t *__restrict__ grad_in, // [nnodes, nfeatures_A, nfeatures_B]
        scalar_t *__restrict__ grad_A,  // [nedges, nfeatures_A]
        scalar_t *__restrict__ grad_B   // [nedges, nfeatures_B]
    ) {

    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;
    const int32_t nThreadRow = blockDim.x / WARP_SIZE;

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_grad_in =
        shared_array<scalar_t>(nfeatures_A * nfeatures_B, sptr, &space);

    scalar_t *buffer_A =
        shared_array<scalar_t>(nThreadRow * nfeatures_A, sptr, &space);
    scalar_t *buffer_B =
        shared_array<scalar_t>(nThreadRow * nfeatures_B, sptr, &space);

    scalar_t *buffer_grad_A =
        shared_array<scalar_t>(nThreadRow * nfeatures_A, sptr, &space);
    scalar_t *buffer_grad_B =
        shared_array<scalar_t>(nThreadRow * WARP_SIZE * TB, sptr, &space);

    /* registers to hold components of A, B and grads - used to increase
     * arithmetic intensity.
     */

    scalar_t regB[TB] = {0.0};
    scalar_t gradB[TB] = {0.0};

    const int32_t edge_start = first_occurences[blockIdx.x];
    const int32_t edge_end = (blockIdx.x == nnodes - 1)
                                 ? nedges_total
                                 : first_occurences[blockIdx.x + 1];
    const int32_t node_index = indices_output[edge_start];
    const int32_t nedges = edge_end - edge_start;

    /* total number of columns of B we can process is TB * WARP_SIZE, so
     * we need to loop find_integer_divisor(nfeatures_B, TB*WARP_SIZE) times
     */

    int32_t niter_B = find_integer_divisor(nfeatures_B, TB * WARP_SIZE);

    /*
     * initialise buffer_grad_in for this sub block
     */

    for (int tid = threadIdx.x; tid < nfeatures_A * nfeatures_B;
         tid += NWARPS_PER_BLOCK * WARP_SIZE) {
        buffer_grad_in[tid] =
            grad_in[node_index * nfeatures_A * nfeatures_B + tid];
    }

    __syncthreads();

    /*
     * buffer_grad_A shape = [NWARPS_PER_BLOCK * nfeatures_A]: need to reduce
     * across warps in a separate step */

    /*
     * buffer_grad_B shape: [nfeatures_B] -> can use warp shuffles to reduce
     * across threads */

    for (int32_t edge_idx = threadRow; edge_idx < nedges;
         edge_idx += nThreadRow) {

        int32_t edge = edge_idx + edge_start;

        /*
         * zero temporary buffers and load A, B into shared memory
         */

        for (int tid = threadIdx.x; tid < nfeatures_A; tid += WARP_SIZE) {
            buffer_grad_A[threadRow * nfeatures_A + tid] = 0.0;
            buffer_A[threadRow * nfeatures_B + tid] =
                A[edge * nfeatures_A + tid];
        }

        for (int tid = threadIdx.x; tid < nfeatures_B; tid += WARP_SIZE) {
            buffer_B[threadRow * nfeatures_B + tid] =
                B[edge * nfeatures_B + tid];
        }

        for (int tid = threadIdx.x; tid < TB * WARP_SIZE; tid += WARP_SIZE) {
            buffer_grad_B[threadRow * (TB * WARP_SIZE) + tid] = 0.0;
        }

        __syncwarp();

        for (int32_t iter_B = 0; iter_B < niter_B; iter_B++) {

            // process the next TB * WARP_SIZE chunk
            int32_t global_B = iter_B * TB * WARP_SIZE;

            /*
             *  load B from SMEM into local registers
             */
            for (int32_t i = 0; i < TB; i++) {
                if (global_B + i * WARP_SIZE + threadCol < nfeatures_A)
                    regB[i] = buffer_B[global_B + i * WARP_SIZE + threadCol];
            }

            /*
             * perform the reduction
             */
            for (int i = 0; i < nfeatures_A; i++) {

                scalar_t dsumA = 0.0;

                for (int j = 0; j < TB; j++) {

                    scalar_t grad_in_ij = 0.0;

                    if (i * nfeatures_B + global_B + j * WARP_SIZE + threadCol <
                        nfeatures_A * nfeatures_B) {
                        grad_in_ij = buffer_grad_in[i * nfeatures_B + global_B +
                                                    j * WARP_SIZE + threadCol];
                    }

                    if (global_B + j * WARP_SIZE + threadCol < nfeatures_B)
                        buffer_grad_B[threadRow * (TB * WARP_SIZE) +
                                      threadCol] += grad_in_ij * buffer_A[i];

                    dsumA += grad_in_ij * regB[j];
                }

                // need to warp shuffle reduce across the threads
                // accessing each B index.
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    dsumA +=
                        __shfl_down_sync(FULL_MASK, dsumA, offset, WARP_SIZE);
                }

                // thread 0 contains the gradient for this subset of features_B.
                if (threadCol == 0)
                    buffer_grad_A[i * nfeatures_A + threadRow] += dsumA;
            }

            __syncwarp();

            // write gradB
            for (int k = threadCol; k < nfeatures_B; k += WARP_SIZE) {
                grad_B[edge * nfeatures_B + threadCol] =
                    buffer_grad_B[threadRow * WARP_SIZE + threadCol];
            }
        }

        __syncwarp();

        // write gradA
        for (int i = threadCol; i < nfeatures_A; i += WARP_SIZE) {
            grad_A[edge * nfeatures_A + threadCol] =
                buffer_grad_A[i * nfeatures_A + threadRow];
        }
    }
}

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_cuda(
    const scalar_t *__restrict__ A, // [nedges, nfeatures_A]
    const scalar_t *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,           // number of nodes we're summing into
    const int32_t nedges,           // number of edges -> batch size of A and B
    const int32_t nfeatures_A,      // number of features of A
    const int32_t nfeatures_B,      // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where
                                        // the values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to
                                                // sum into [nedges]
    scalar_t *__restrict__ output // shape: [nnodes, nfeatures_B, nfeatures_A]
                                  // -> this ordering because contiguity of
                                  // threadCol

) {

    dim3 gridDim(nnodes, 1, 1);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    outer_product_scatter_add_kernel<scalar_t, 2, 2><<<gridDim, blockDim, 0>>>(
        A, B, nnodes, nedges, nfeatures_A, nfeatures_B, first_occurences,
        indices_output, output);

    cudaDeviceSynchronize();
}

template <>
void mops::cuda::outer_product_scatter_add_cuda<float>(
    const float *__restrict__ A, // [nedges, nfeatures_A]
    const float *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,        // number of nodes we're summing into
    const int32_t nedges,        // number of edges -> batch size of A and B
    const int32_t nfeatures_A,   // number of features of A
    const int32_t nfeatures_B,   // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where
                                        // the values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to
                                                // sum into [nedges]
    float *__restrict__ output // shape: [nnodes, nfeatures_B, nfeatures_A]
                               // -> this ordering because contiguity of
                               // threadCol
);

template <>
void mops::cuda::outer_product_scatter_add_cuda<double>(
    const double *__restrict__ A, // [nedges, nfeatures_A]
    const double *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,         // number of nodes we're summing into
    const int32_t nedges,         // number of edges -> batch size of A and B
    const int32_t nfeatures_A,    // number of features of A
    const int32_t nfeatures_B,    // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where
                                        // the values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to
                                                // sum into [nedges]
    double *__restrict__ output // shape: [nnodes, nfeatures_B, nfeatures_A]
                                // -> this ordering because contiguity of
                                // threadCol
);

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_vjp_cuda(
    const scalar_t *__restrict__ A, // [nedges, nfeatures_A]
    const scalar_t *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,           // number of nodes we're summing into
    const int32_t nedges,           // number of edges -> batch size of A and B
    const int32_t nfeatures_A,      // number of features of A
    const int32_t nfeatures_B,      // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where
                                        // the values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to
                                                // sum into [nedges]
    scalar_t *__restrict__ grad_in, // grad_input: [nnodes, nfeatures_B,
                                    // nfeatures_A]
    scalar_t *__restrict__ grad_A,  // [nedges, nfeatures_A],
    scalar_t *__restrict__ grad_B   // [nedges, nfeatures_B]
) {

    dim3 gridDim(nnodes, 1, 1);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    void *sptr = 0;
    size_t space = 0;

    shared_array<scalar_t>(nfeatures_A * nfeatures_B, sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * nfeatures_A, sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * nfeatures_B, sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * nfeatures_A, sptr, &space);
    shared_array<scalar_t>(NWARPS_PER_BLOCK * WARP_SIZE * 2, sptr, &space);

    outer_product_scatter_add_vjp_kernel<scalar_t, 2>
        <<<gridDim, blockDim, space>>>(A, B, nnodes, nedges, nfeatures_A,
                                       nfeatures_B, first_occurences,
                                       indices_output, grad_in, grad_A, grad_B);

    cudaDeviceSynchronize();
}

template <>
void mops::cuda::outer_product_scatter_add_vjp_cuda<float>(
    const float *__restrict__ A, // [nedges, nfeatures_A]
    const float *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,        // number of nodes we're summing into
    const int32_t nedges,        // number of edges -> batch size of A and B
    const int32_t nfeatures_A,   // number of features of A
    const int32_t nfeatures_B,   // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where
                                        // the values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to
                                                // sum into [nedges]
    float *__restrict__ grad_in, // grad_input: [nnodes, nfeatures_B,
                                 // nfeatures_A]
    float *__restrict__ grad_A,  // [nedges, nfeatures_A],
    float *__restrict__ grad_B   // [nedges, nfeatures_B]
);

template <>
void mops::cuda::outer_product_scatter_add_vjp_cuda<double>(
    const double *__restrict__ A, // [nedges, nfeatures_A]
    const double *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,         // number of nodes we're summing into
    const int32_t nedges,         // number of edges -> batch size of A and B
    const int32_t nfeatures_A,    // number of features of A
    const int32_t nfeatures_B,    // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where
                                        // the values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to
                                                // sum into [nedges]
    double *__restrict__ grad_in, // grad_input: [nnodes, nfeatures_B,
                                  // nfeatures_A]
    double *__restrict__ grad_A,  // [nedges, nfeatures_A],
    double *__restrict__ grad_B   // [nedges, nfeatures_B]
);
