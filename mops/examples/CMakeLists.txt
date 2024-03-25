cmake_minimum_required(VERSION 3.16)

add_executable(example_cpp example.cpp)
target_link_libraries(example_cpp mops)
add_test(NAME example_cpp COMMAND ./example_cpp)

if (CMAKE_CUDA_COMPILER)
    add_executable(example_cuda example.cu)
    target_link_libraries(example_cuda mops)
    add_test(NAME example_cuda COMMAND ./example_cuda)
endif()
