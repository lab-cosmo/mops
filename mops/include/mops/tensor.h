#ifndef MOPS_TENSOR_H
#define MOPS_TENSOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mops_tensor_3d_f32_t {
    float *MOPS_RESTRICT data;
    int64_t shape[3];
};

struct mops_tensor_3d_f64_t {
    double *MOPS_RESTRICT data;
    int64_t shape[3];
};

struct mops_tensor_2d_f32_t {
    float *MOPS_RESTRICT data;
    int64_t shape[2];
};

struct mops_tensor_2d_f64_t {
    double *MOPS_RESTRICT data;
    int64_t shape[2];
};

struct mops_tensor_1d_f32_t {
    float *MOPS_RESTRICT data;
    int64_t shape[1];
};

struct mops_tensor_1d_f64_t {
    double *MOPS_RESTRICT data;
    int64_t shape[1];
};

struct mops_tensor_1d_i32_t {
    int32_t *MOPS_RESTRICT data;
    int64_t shape[1];
};

struct mops_tensor_2d_i32_t {
    int32_t *MOPS_RESTRICT data;
    int64_t shape[2];
};

#ifdef __cplusplus
}
#endif

#endif
