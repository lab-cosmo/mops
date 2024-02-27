#ifndef FIRST_OCCURENCES_HPP
#define FIRST_OCCURENCES_HPP

#include "mops/tensor.hpp"
#include <cstdint>

namespace mops {

namespace cuda {
int *calculate_first_occurences_cuda(const int32_t *receiver_list,
                                     int32_t nedges, int32_t nnodes);

} // namespace cuda
} // namespace mops

#endif // FIRST_OCCURENCES_HP