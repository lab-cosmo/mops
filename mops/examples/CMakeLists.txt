cmake_minimum_required(VERSION 3.16)

add_executable(example_cpp example.cpp)
target_link_libraries(example_cpp mops)
add_test(NAME example_cpp COMMAND ./example_cpp)
