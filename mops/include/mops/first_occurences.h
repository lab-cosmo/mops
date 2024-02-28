#ifndef MOPS_FIRST_OCCURENCES_H
#define MOPS_FIRST_OCCURENCES_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CUDA version of mops::cuda::first_occurences
MOPS_EXPORT int *mops_cuda_first_occurences(mops_tensor_1d_i32_t receiver_list,
                                            int32_t nedges, int32_t natoms);

#ifdef __cplusplus
}
#endif

#endif
