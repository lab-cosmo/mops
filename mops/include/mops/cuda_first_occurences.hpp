#ifndef FIRST_OCCURENCES_HPP
#define FIRST_OCCURENCES_HPP

#include "mops/tensor.hpp"
#include <cstdint>

int *calculate_first_occurences_cuda(const int32_t *receiver_list,
                                     int32_t nedges, int32_t nnodes);

#endif // FIRST_OCCURENCES_HP