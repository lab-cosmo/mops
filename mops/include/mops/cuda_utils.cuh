#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cstdint>
#include <cuda.h>

using namespace std;

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim);

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr,
                                    std::size_t *space) noexcept;

template <class T>
__host__ __device__ T *align_array(std::size_t n_elements, void *&ptr,
                                   const std::size_t alignment,
                                   std::size_t *space) noexcept;

__host__ int32_t *
calculate_first_occurences_cuda(const int32_t *__restrict__ receiver_list,
                                int32_t nedges, int32_t natoms,
                                int32_t *__restrict__ first_occurences);

#endif // CUDA_UTILS_CUH