#ifndef FIRST_OCCURENCES_HPP
#define FIRST_OCCURENCES_HPP

#include <cstdint>

namespace mops {

namespace cuda {
int32_t *
calculate_first_occurences_cuda(const int32_t *__restrict__ receiver_list,
                                int32_t nedges, int32_t natoms,
                                int32_t *__restrict__ first_occurences);

} // namespace cuda
} // namespace mops

#endif // FIRST_OCCURENCES_HP