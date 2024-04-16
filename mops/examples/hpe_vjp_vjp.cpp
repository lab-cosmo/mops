#include "mops.hpp"
#include <vector>

int main() {

    auto grad_grad_output = std::vector<double>(100);
    auto grad_A_2 = std::vector<double>(100 * 100);
    auto grad_grad_A = std::vector<double>(100 * 100);
    auto grad_output = std::vector<double>(100);
    auto A = std::vector<double>(100 * 100);
    auto C = std::vector<double>(100);
    auto indices_A = std::vector<int32_t>(100 * 4);

    mops::homogeneous_polynomial_evaluation_vjp_vjp<double>(
        {grad_grad_output.data(), {100}},
        {grad_A_2.data(), {100, 100}},
        {grad_grad_A.data(), {100, 100}},
        {grad_output.data(), {100}},
        {A.data(), {100, 100}},
        {C.data(), {100}},
        {indices_A.data(), {100, 4}}
    );

    return 0;
}
