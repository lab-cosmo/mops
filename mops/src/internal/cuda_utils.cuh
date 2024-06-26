#ifndef MOPS_CUDA_UTILS_CUH
#define MOPS_CUDA_UTILS_CUH

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

/*
 * Pre SM60 cards do not support atomicAdd(double *, double). This function implements and atomicCAS
 * to lock update the address.
 */
__device__ double atomicAdd_presm60(double* address, double val);

/*
 * function to select the right version of atomicAdd for the archcode being compiled.
 */
template <typename scalar_t> __device__ scalar_t ATOMIC_ADD(scalar_t* address, scalar_t val);

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim);

/*
 * helper function to allocate correctly sized shared memory buffers. creates a pointer reference to
 * a shared memory array with n_elements number of elements. On exit, ptr is shifted right by
 * nelements * sizeof(T), and space is incremented by this same amount.
 *
 */
template <class T>
__host__ __device__ T* shared_array(std::size_t n_elements, void*& ptr, std::size_t* space) noexcept;

/*
 * helper function to allocate correctly sized shared memory buffers identically to the shared_array
 *  method, but in addition are aligned to a certain byte boundary given by alignment.
 */
template <class T>
__host__ __device__ T* align_array(
    std::size_t n_elements, void*& ptr, const std::size_t alignment, std::size_t* space
) noexcept;

#endif // CUDA_UTILS_CUH
