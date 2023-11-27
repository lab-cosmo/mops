#include <chrono>
#include <vector>
#include <iostream>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(60000 * 20);
    fill_vector_random_floats(B);

    auto W = std::vector<double>(1000 * 20);
    fill_vector_random_floats(W);

    auto indices_W = std::vector<int>(60000);
    fill_vector_random_integers(indices_W, 1000);

    auto indices_output = std::vector<int>(60000);
    fill_vector_random_integers(indices_output, 1000);

    auto O = std::vector<double>(1000 * 13 * 20);

    std::vector<double> execution_times(1000);
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::outer_product_scatter_add_with_weights<double>(
            {O.data(), {1000, 13, 20}}, {A.data(), {60000, 13}},
            {B.data(), {60000, 20}}, {W.data(), {1000, 20}},
            {indices_W.data(), {60000}}, {indices_output.data(), {60000}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean, stddev] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time: " << mean << " ms\n";
    std::cout << "Standard Deviation: " << stddev << " ms\n";

    auto grad_output = std::vector<double>(1000 * 13 * 20);
    fill_vector_random_floats(grad_output);

    auto grad_A = std::vector<double>(60000 * 13);
    auto grad_B = std::vector<double>(60000 * 20);
    auto grad_W = std::vector<double>(1000 * 20);

    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::outer_product_scatter_add_with_weights_vjp<double>(
            {grad_A.data(), {60000, 13}}, {grad_B.data(), {60000, 20}},
            {grad_W.data(), {1000, 20}}, {grad_output.data(), {1000, 13, 20}},
            {A.data(), {60000, 13}}, {B.data(), {60000, 20}},
            {W.data(), {1000, 20}}, {indices_W.data(), {60000}},
            {indices_output.data(), {60000}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean_vjp, stddev_vjp] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time vjp: " << mean_vjp << " ms\n";
    std::cout << "Standard Deviation vjp: " << stddev_vjp << " ms\n";

    return 0;
}
