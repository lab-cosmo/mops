#ifndef MOPS_CUDA_FIRST_OCCURENCES_HPP
#define MOPS_CUDA_FIRST_OCCURENCES_HPP

#include <cstdint>

/*
 * Computes the indexes at which the sorted input array (receiver_list) change in value, with 0
 * prepended. For example, if the receiver list is [0, 0, 0, 1, 1, 1, 2, 2], then the output would
 * be [0, 3, 6]. nelements_input refers to the size of the input receiver_list, nelements_output
 * refers, to the number of output elements. The elements of receiver_list **must** be sorted such
 * that all references to each index appear contiguously and continuously.
 *
 * This function allocates memory which must be freed with `cudaFree` when no longer required.
 */
int32_t* calculate_first_occurences_cuda(
    const int32_t* receiver_list, int32_t nelements_input, int32_t nelements_output
);

#endif // MOPS_CUDA_FIRST_OCCURENCES_HPP
