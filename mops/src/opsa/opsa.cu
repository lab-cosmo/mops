
#include "mops/cuda_utils.cuh"
#include "mops/opsa_cuda.cuh"

using namespace mops::cuda;

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

template <typename scalar_t, const int32_t TA, const int32_t TB>
__device__ void outer_product_scatter_add_kernel(
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
             * [node, nfeatures_A, nfeatures_B]
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
namespace mops::cuda {
template <typename scalar_t>
void outer_product_scatter_add_cuda(
    const scalar_t *__restrict__ A, // [nedges, nfeatures_A]
    const scalar_t *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,           // number of nodes we're summing into
    const int32_t nedges,           // number of edges -> batch size of A and B
    const int32_t nfeatures_A,      // number of features of A
    const int32_t nfeatures_B,      // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where the
                                        // values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to sum
                                                // into [nedges]
    scalar_t
        *__restrict__ output // shape: [nnodes, nfeatures_B, nfeatures_A]
                             // -> this ordering because contiguity of threadCol

) {

    dim3 gridDim(nnodes, 1, 1);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    outer_product_scatter_add_kernel<scalar_t, 4, 4><<<gridDim, blockDim, 0>>>(
        A, B, nnodes, nedges, nfeatures_A, nfeatures_B, first_occurences,
        indices_output, output);

    cudaDeviceSynchronize();
}

} // namespace mops::cuda
