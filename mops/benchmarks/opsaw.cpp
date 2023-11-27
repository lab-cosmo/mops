#include <chrono>
#include <vector>
#include <iostream>

#include "mops.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    size_t output_size = 1000;
    if (argc > 2) {
        std::cout << "This program only takes one command-line argument\n";
        return 1;
    } else if (argc == 2) {
        output_size = std::stoi(argv[1]);
    } else {
        // Use default value of 1000
    }
    size_t input_size = 60 * output_size;

    auto A = std::vector<double>(input_size * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(input_size * 20);
    fill_vector_random_floats(B);

    auto W = std::vector<double>(output_size * 20);
    fill_vector_random_floats(W);

    auto indices_W = std::vector<int>(input_size);
    fill_vector_random_integers(indices_W, output_size);

    auto indices_output = std::vector<int>(input_size);
    fill_vector_random_integers(indices_output, output_size);

    auto O = std::vector<double>(output_size * 13 * 20);

    std::vector<double> execution_times(100);
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::outer_product_scatter_add_with_weights<double>(
            {O.data(), {output_size, 13, 20}}, {A.data(), {input_size, 13}},
            {B.data(), {input_size, 20}}, {W.data(), {output_size, 20}},
            {indices_W.data(), {input_size}}, {indices_output.data(), {input_size}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean, stddev] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time: " << mean << " ms\n";
    std::cout << "Standard Deviation: " << stddev << " ms\n";

    auto grad_output = std::vector<double>(output_size * 13 * 20);
    fill_vector_random_floats(grad_output);

    auto grad_A = std::vector<double>(input_size * 13);
    auto grad_B = std::vector<double>(input_size * 20);
    auto grad_W = std::vector<double>(output_size * 20);

    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::outer_product_scatter_add_with_weights_vjp<double>(
            {grad_A.data(), {input_size, 13}}, {grad_B.data(), {input_size, 20}},
            {grad_W.data(), {output_size, 20}}, {grad_output.data(), {output_size, 13, 20}},
            {A.data(), {input_size, 13}}, {B.data(), {input_size, 20}},
            {W.data(), {output_size, 20}}, {indices_W.data(), {input_size}},
            {indices_output.data(), {input_size}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean_vjp, stddev_vjp] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time vjp: " << mean_vjp << " ms\n";
    std::cout << "Standard Deviation vjp: " << stddev_vjp << " ms\n";

    return 0;
}
