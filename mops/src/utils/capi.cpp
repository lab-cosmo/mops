#include "mops/capi.hpp"

#include "mops/cuda_first_occurences.hpp"
#include "mops/first_occurences.h"

static size_t checked_cast(int64_t value) {
    if (value < 0 ||
        static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error("integer value '" + std::to_string(value) +
                                 "' does not fit in size_t");
    }
    return static_cast<size_t>(value);
}

extern "C" int32_t *mops_cuda_first_occurences(mops_tensor_1d_i32_t receiver_list,
                                           int32_t nedges, int32_t nnodes) {

    MOPS_CATCH_EXCEPTIONS_WITH_RETURN(
        mops::cuda::calculate_first_occurences_cuda(receiver_list.data, nedges,
                                                    nnodes));
}