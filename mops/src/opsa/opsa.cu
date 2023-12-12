// todo: cuda device code
#define WARP_SIZE 32

/* these two functions should go in a utility code.*/
__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim) {
    return (x + bdim - 1) / bdim;
}

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr,
                                    std::size_t *space = nullptr) noexcept {
    const std::uintptr_t inptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t end = inptr + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - inptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(inptr);
}

template <typename scalar_t, const int TA, const int TB>
__device__ void outer_product_scatter_add_kernel(
    const scalar_t *__restrict__ A, const scalar_t *__restrict__ B,
    const int nnodes,      // number of nodes we're summing into
    const int nedges,      // number of edges -> batch size of A and B
    const int nfeatures_A, // number of features of A [nedges, nfeatures_A]
    const int nfeatures_B, // number of features of B [nedges, nfeatures_B]
    const int32_t
        *__restrict__ first_occurences, // indices in indices_output where the
                                        // values change [nnodes]
    const int32_t *__restrict__ indices_output, // sorted list of indices to sum
                                                // into [nedges]
    scalar_t *__restrict__ output // shape: [nnodes, nfeatures_B, nfeatures_A]
                                  // -> this ordering because threadCol % 32
) {

    extern __shared__ char buffer[];

    const uint threadCol = threadIdx.x % WARP_SIZE;
    const uint threadRow = thredIdx.x / WARP_SIZE;
    const uint nThreadRow = blockDim.x / WARP_SIZE;

    void *sptr = buffer;
    size_t space = 0;

    /*
     * Shared memory buffers to alleviate MIO stalls.
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

    const uint edge_start = first_occurences[blockIdx.x];
    const uint edge_end =
        (blockIdx.x == nnodes - 1) ? nedges : first_occurences[blockIdx.x + 1];
    const uint node_index = indices_output[edge_start];
    const uint nedges = edge_end - edge_start;

    /* total number of columns of A we can process is TA * WARP_SIZE, so
     * we need to loop find_integer_divisor(nfeatures_A, TA*WARP_SIZE) times
     */

    int niter_A = find_integer_divisor(nfeatures_A, TA * WARP_SIZE);
    int niter_B = find_integer_divisor(nfeatures_B, TB * nThreadRow);

    for (uint iter_B = 0; iter_B < niter_B; iter_B++) {
        for (uint iter_A = 0; iter_A < niter_A; iter_A++) {

            /*
             *  clear registers
             */
            for (uint i = 0; i < TA; i++) {
                regA[i] = 0;
            }

            for (uint i = 0; i < TB; i++) {
                regB[i] = 0;
            }

            for (uint i = 0; i < TA * TB; i++) {
                regOP[i] = 0.0;
            }

            for (uint edge = 0; edge < nedges; edge++) {
                /*
                 *  load A from GMEM into local registers
                 */
                for (uint i = 0; i < TA; i++) {

                    if ((iter_A * TA * WARP_SIZE) + i * WARP_SIZE + threadCol <
                        nfeatures_A)
                        regA[i] = A[(edge_start + edge) * nfeatures_A +
                                    (iter_A * TA * WARP_SIZE) + i * WARP_SIZE +
                                    threadCol];
                }

                /*
                 *  load B from GMEM into local registers
                 */
                for (uint i = 0; i < TB; i++) {
                    if ((iter_B * TB * nThreadRow) + i * nThreadRow +
                            threadRow <
                        nfeatures_B)
                        regB[i] = B[(edge_start + edge) * nfeatures_B +
                                    (iter_B * TB * nThreadRow) +
                                    i * nThreadRow + threadRow];
                }

                /*
                 * perform outer product in registers
                 */
                for (int j = 0; j < TB; j++) {
                    for (int i = 0; i < TA; i++) {
                        regOP[j * TA + i] += regA[i] * regB[j];
                    }
                }
            }

            /*
             * writeout the content of regOP to the output for this block of
             * [node, nfeatures_A, nfeatures_B]
             */
            for (int i = 0; i < TA; i++) {
                for (int j = 0; j < TB; j++) {
                    if ((iter_A * TA * WARP_SIZE) + i * WARP_SIZE + threadCol <
                            nfeatures_A &&
                        (iter_B * TB * nThreadRow) + i * nThreadRow +
                                threadRow <
                            nfeatures_B) {
                        output[node_index * nfeatures_B * nfeatures_A +
                               ((iter_B * TB * nThreadRow) + i * nThreadRow +
                                threadRow) *
                                   nfeatures_A +
                               (iter_A * TA * WARP_SIZE) + i * WARP_SIZE +
                               threadCol] = regOP[i * TB + j];
                    }
                }
            }
        }
    }
}