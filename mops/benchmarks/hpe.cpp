#include <vector>
#include <chrono>
#include <iostream>

#include "mops.hpp"
#include "utils.hpp"

int main() {
    auto A = std::vector<double>(1000 * 2000);
    fill_vector_random_floats(A);

    auto C = std::vector<double>(100000);
    fill_vector_random_floats(C);

    auto indices_A = std::vector<int>(100000 * 4);
    fill_vector_random_integers(indices_A, 2000);

    auto output = std::vector<double>(1000);

    std::vector<double> execution_times(1000);
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::homogeneous_polynomial_evaluation<double>(
            {output.data(), {1000}}, {A.data(), {1000, 2000}},
            {C.data(), {100000}}, {indices_A.data(), {100000, 4}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean, stddev] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time: " << mean << " ms\n";
    std::cout << "Standard Deviation: " << stddev << " ms\n";


    auto grad_output = std::vector<double>(1000);
    fill_vector_random_floats(grad_output);

    auto grad_A = std::vector<double>(1000 * 2000);

    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::homogeneous_polynomial_evaluation_vjp<double>(
            {grad_A.data(), {1000, 2000}}, {grad_output.data(), {1000}}, {A.data(), {1000, 2000}},
            {C.data(), {100000}}, {indices_A.data(), {100000, 4}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean_vjp, stddev_vjp] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time vjp: " << mean_vjp << " ms\n";
    std::cout << "Standard Deviation vjp: " << stddev_vjp << " ms\n";

    return 0;
}
