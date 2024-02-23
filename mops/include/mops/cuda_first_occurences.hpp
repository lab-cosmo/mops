#ifndef FIRST_OCCURENCES_HPP
#define FIRST_OCCURENCES_HPP

#include "mops/tensor.hpp"
#include <cstdint>

namespace mops {

namespace cuda {
int32_t *calculate_first_occurences_cuda(int32_t *receiver_list, int32_t nedges,
                                         int32_t natoms,
                                         int32_t *first_occurences);

} // namespace cuda
} // namespace mops

#endif // FIRST_OCCURENCES_HP