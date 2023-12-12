// todo: cuda device code
#define WARP_SIZE 32

/* these two functions should go in a utility code.*/
__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim) {
    return (x + bdim - 1) / bdim;
}

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr,
                                    std::size_t *space = nullptr) noexcept {
    const std::int32_tptr_t inptr = reinterpret_cast<int32_tptr_t>(ptr);
    const std::int32_tptr_t end = inptr + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - inptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(inptr);
}

template <typename scalar_t, const int32_t TA, const int32_t TB>
__device__ void outer_product_scatter_add_kernel(
    const scalar_t *__restrict__ A, // [nedges, nfeatures_A]
    const scalar_t *__restrict__ B, // [nedges, nfeatures_B]
    const int32_t nnodes,               // number of nodes we're summing into
    const int32_t nedges,               // number of edges -> batch size of A and B
    const int32_t nfeatures_A,          // number of features of A
    const int32_t nfeatures_B,          // number of features of B
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where the
                                        // values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to sum
                                                // into [nedges]
    scalar_t
        *__restrict__ output // shape: [nnodes, nfeatures_B, nfeatures_A]
                             // -> this ordering because contiguity of threadCol
) {

    extern __shared__ char buffer[];

    const int32_t threadCol = threadIdx.x % WARP_SIZE;
    const int32_t threadRow = threadIdx.x / WARP_SIZE;
    const int32_t nThreadRow = blockDim.x / WARP_SIZE;

    void *sptr = buffer;
    size_t space = 0;

    /*
     * Shared memory buffers to alleviate MIO stalls -> implement double buffering + async memcopies for Ampere +?
     * pipeline could be GMEM -> SMEM -> registers -> compute

    scalar_t *buffer_A = shared_array<scalar_t>(TA * WARP_SIZE, sptr,
    &space); scalar_t *buffer_B = shared_array<scalar_t>(TB * nThreadRow,
    sptr, &space);

     */

    /* registers to hold components of A, B and output - used to increase
     * arithmetic intensity.
     */
    scalar_t regA[TA] = {0.0};
    scalar_t regB[TB] = {0.0};
    scalar_t regOP[TA * TB] = {0.0};

    const int32_t edge_start = first_occurences[blockIdx.x];
    const int32_t edge_end =
        (blockIdx.x == nnodes - 1) ? nedges : first_occurences[blockIdx.x + 1];
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

            for (int32_t edge = 0; edge < nedges; edge++) {
                /*
                 *  load A from GMEM into local registers
                 */
                for (int32_t i = 0; i < TA; i++) {

                    if (global_A + i * WARP_SIZE + threadCol < nfeatures_A)
                        regA[i] = A[(edge_start + edge) * nfeatures_A +
                                    global_A + i * WARP_SIZE + threadCol];
                }

                /*
                 *  load B from GMEM into local registers
                 */
                for (int32_t i = 0; i < TB; i++) {
                    if (global_B + i * nThreadRow + threadRow < nfeatures_B)
                        regB[i] = B[(edge_start + edge) * nfeatures_B +
                                    global_B + i * nThreadRow + threadRow];
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