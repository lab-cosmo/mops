#include <vector>
#include "mops.hpp"


int main() {

    auto A = std::vector<double>(100*100);
    auto C = std::vector<double>(100);
    auto P = std::vector<int>(100*4);
    auto O = std::vector<double>(100);

    mops::homogeneous_polynomial_evaluation<double>(
        {O.data(), {100}},
        {A.data(), {100, 100}},
        {C.data(), {100}},
        {P.data(), {100, 4}}
    );

    return 0;
}
