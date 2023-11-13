cmake_minimum_required(VERSION 3.16)

add_executable(sap sap.cpp)
target_link_libraries(sap mops)
add_test(NAME sap COMMAND ./sap)
