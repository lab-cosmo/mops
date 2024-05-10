#include <cstdint>
#include <cstdio>

#include <cuda.h>

#include "cuda_utils.cuh"

using namespace std;

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim) {
    return (x + bdim - 1) / bdim;
}

__device__ double atomicAdd_presm60(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))
        );

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

template <typename scalar_t> __device__ scalar_t ATOMIC_ADD(scalar_t* address, scalar_t val) {
#if __CUDA_ARCH__ < 600
    if constexpr (sizeof(scalar_t) == 4) {
        return atomicAdd(address, val);
    } else if constexpr (sizeof(scalar_t) == 8) {
        return atomicAdd_presm60(address, val);
    }
#else
    return atomicAdd(address, val);
#endif
}

template float ATOMIC_ADD<float>(float* address, float val);
template double ATOMIC_ADD<double>(double* address, double val);

template <typename T>
__host__ __device__ T* shared_array(std::size_t n_elements, void*& ptr, std::size_t* space) noexcept {
    const std::uintptr_t inptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t end = inptr + n_elements * sizeof(T);
    if (space) {
        *space += static_cast<std::size_t>(end - inptr);
    }
    ptr = reinterpret_cast<void*>(end);
    return reinterpret_cast<T*>(inptr);
}

// forward declare for different types...
template float* shared_array<float>(std::size_t n_elements, void*& ptr, std::size_t* space) noexcept;
template double* shared_array<double>(std::size_t n_elements, void*& ptr, std::size_t* space) noexcept;
template int* shared_array<int>(std::size_t n_elements, void*& ptr, std::size_t* space) noexcept;
template short* shared_array<short>(std::size_t n_elements, void*& ptr, std::size_t* space) noexcept;

template <typename T>
__host__ __device__ T* align_array(
    std::size_t n_elements, void*& ptr, const std::size_t alignment, std::size_t* space
) noexcept {
    const std::uintptr_t intptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t aligned = (intptr + alignment - 1) & -alignment;
    const std::uintptr_t end = aligned + n_elements * sizeof(T);
    if (space) {
        *space += static_cast<std::size_t>(end - intptr);
    }
    ptr = reinterpret_cast<void*>(end);
    return reinterpret_cast<T*>(aligned);
}

// forward declare for different types...
template float* align_array<float>(
    std::size_t n_elements, void*& ptr, const std::size_t alignment, std::size_t* space
) noexcept;
template double* align_array<double>(
    std::size_t n_elements, void*& ptr, const std::size_t alignment, std::size_t* space
) noexcept;
template int* align_array<int>(
    std::size_t n_elements, void*& ptr, const std::size_t alignment, std::size_t* space
) noexcept;
template short* align_array<short>(
    std::size_t n_elements, void*& ptr, const std::size_t alignment, std::size_t* space
) noexcept;
