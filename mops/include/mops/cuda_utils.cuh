#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cstdint>
#include <cstdio>
#include <cuda.h>

using namespace std;

#define CUDA_CHECK_ERROR(err)                                                                      \
    do {                                                                                           \
        cudaError_t err_cuda = err;                                                                \
        if (err_cuda != cudaSuccess) {                                                             \
            fprintf(                                                                               \
                stderr,                                                                            \
                "CUDA error in file '%s' in line %i: %s\n",                                        \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                cudaGetErrorString(err_cuda)                                                       \
            );                                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim);

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr, std::size_t *space) noexcept;

template <class T>
__host__ __device__ T *align_array(
    std::size_t n_elements, void *&ptr, const std::size_t alignment, std::size_t *space
) noexcept;

#endif // CUDA_UTILS_CUH