#include "mops.hpp"
#include <vector>

int main() {

    auto A = std::vector<double>(100 * 100);
    auto C = std::vector<double>(100);
    auto indices_A = std::vector<int>(100 * 4);
    auto output = std::vector<double>(100);

    mops::homogeneous_polynomial_evaluation<double>(
        {output.data(), {100}}, {A.data(), {100, 100}}, {C.data(), {100}},
        {indices_A.data(), {100, 4}});

    return 0;
}
