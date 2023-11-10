#ifndef MOPS_TENSOR_H
#define MOPS_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mops_tensor_3d_f32_t {
    float* __restrict__ data;
    int64_t shape[3];
};

struct mops_tensor_3d_f64_t {
    double* __restrict__ data;
    int64_t shape[3];
};

struct mops_tensor_2d_f32_t {
    float* __restrict__ data;
    int64_t shape[2];
};

struct mops_tensor_2d_f64_t {
    double* __restrict__ data;
    int64_t shape[2];
};

struct mops_tensor_1d_f32_t {
    float* __restrict__ data;
    int64_t shape[1];
};

struct mops_tensor_1d_f64_t {
    double* __restrict__ data;
    int64_t shape[1];
};

struct mops_tensor_1d_i32_t {
    int32_t* __restrict__ data;
    int64_t shape[1];
};

struct mops_tensor_2d_i32_t {
    int32_t* __restrict__ data;
    int64_t shape[2];
};

#ifdef __cplusplus
}
#endif


#endif
