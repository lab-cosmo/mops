#include "mops.hpp"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/*host macro that checks for errors in CUDA calls, and prints the file + line
 * and error string if one occurs
 */
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t cudaStatus = (call);                                                           \
        if (cudaStatus != cudaSuccess) {                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "                  \
                      << cudaGetErrorString(cudaStatus) << std::endl;                              \
            cudaDeviceReset();                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

int main() {
    // To avoid calls with a very large number of arguments,
    // mops uses a mops::Tensor<T, N_DIMS> struct which simply
    // consists a data pointer and a shape in the form of a std::array.
    //
    // All mops operations take mops::Tensor objects as their
    // inputs, and these can be initialized in the following way:

    auto A = std::vector<double>(100 * 20);
    auto B = std::vector<double>(100 * 5);
    auto indices_output = std::vector<int32_t>(100);
    auto output = std::vector<double>(10 * 20 * 5);

    double* A_cuda;
    double* B_cuda;
    int32_t* indices_output_cuda;
    double* output_cuda;

    CUDA_CHECK(cudaMalloc(&A_cuda, A.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&B_cuda, B.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&indices_output_cuda, indices_output.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&output_cuda, output.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(A_cuda, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_cuda, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        indices_output_cuda,
        indices_output.data(),
        indices_output.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        output_cuda, output.data(), output.size() * sizeof(double), cudaMemcpyHostToDevice
    ));

    mops::cuda::outer_product_scatter_add<double>(
        {output_cuda, {100, 20, 5}},
        {A_cuda, {100, 20}},
        {B_cuda, {100, 5}},
        {indices_output_cuda, {100}}
    );

    CUDA_CHECK(cudaMemcpy(
        output.data(), output_cuda, output.size() * sizeof(double), cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(A_cuda));
    CUDA_CHECK(cudaFree(B_cuda));
    CUDA_CHECK(cudaFree(indices_output_cuda));
    CUDA_CHECK(cudaFree(output_cuda));

    return 0;
}
