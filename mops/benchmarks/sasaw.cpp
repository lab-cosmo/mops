#include <iostream>
#include <chrono>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto output = std::vector<double>(1000 * 100 * 32);
    fill_vector_random_floats(output);

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(60000 * 32);
    fill_vector_random_floats(B);

    auto C = std::vector<double>(900);
    fill_vector_random_floats(C);

    auto W = std::vector<double>(1000 * 7 * 32);
    fill_vector_random_floats(A);

    auto indices_A = std::vector<int>(900);
    fill_vector_random_integers(indices_A, 13);

    auto indices_W_1 = std::vector<int>(60000);
    fill_vector_random_integers(indices_W_1, 1000);

    auto indices_W_2 = std::vector<int>(900);
    fill_vector_random_integers(indices_W_2, 7);

    auto indices_output_1 = std::vector<int>(60000);
    fill_vector_random_integers(indices_output_1, 1000);

    auto indices_output_2 = std::vector<int>(900);
    fill_vector_random_integers(indices_output_2, 100);

    auto execution_times = std::vector<double>(100);
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::sparse_accumulation_scatter_add_with_weights<double>(
            {output.data(), {1000, 100, 32}}, {A.data(), {60000, 13}},
            {B.data(), {60000, 32}}, {C.data(), {900}},
            {W.data(), {1000, 7, 32}}, {indices_A.data(), {900}},
            {indices_W_1.data(), {60000}}, {indices_W_2.data(), {900}},
            {indices_output_1.data(), {60000}},
            {indices_output_2.data(), {900}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean, stddev] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time: " << mean << " ms\n";
    std::cout << "Standard Deviation: " << stddev << " ms\n";

    auto grad_output = std::vector<double>(1000 * 100 * 32);
    fill_vector_random_floats(grad_output);

    auto grad_A = std::vector<double>(60000 * 13);
    auto grad_B = std::vector<double>(60000 * 32);
    auto grad_W = std::vector<double>(1000 * 7 * 32);

    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::sparse_accumulation_scatter_add_with_weights_vjp<double>(
            {grad_A.data(), {60000, 13}}, {grad_B.data(), {60000, 32}},
            {grad_W.data(), {1000, 7, 32}}, {grad_output.data(), {1000, 100, 32}},
            {A.data(), {60000, 13}}, {B.data(), {60000, 32}},
            {C.data(), {900}}, {W.data(), {1000, 7, 32}},
            {indices_A.data(), {900}}, {indices_W_1.data(), {60000}},
            {indices_W_2.data(), {900}}, {indices_output_1.data(), {60000}},
            {indices_output_2.data(), {900}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean_vjp, stddev_vjp] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time vjp: " << mean_vjp << " ms\n";
    std::cout << "Standard Deviation vjp: " << stddev_vjp << " ms\n";

    return 0;
}
