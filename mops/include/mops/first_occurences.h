#ifndef MOPS_OPSA_H
#define MOPS_OPSA_H

#include "mops/exports.h"
#include "mops/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CUDA version of mops::cuda::first_occurences
int MOPS_EXPORT mops_cuda_first_occurences(mops_tensor_1d_i32_t receiver_list,
                                           int32_t nedges, int32_t natoms,
                                           mops_tensor_1d_i32_t output);

#ifdef __cplusplus
}
#endif

#endif
