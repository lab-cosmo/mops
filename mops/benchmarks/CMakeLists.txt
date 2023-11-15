cmake_minimum_required(VERSION 3.16)

add_executable(sap sap.cpp utils.hpp)
target_link_libraries(sap mops)
add_test(NAME sap COMMAND ./sap)

add_executable(hpe hpe.cpp utils.hpp)
target_link_libraries(hpe mops)
add_test(NAME hpe COMMAND ./hpe)

add_executable(sasax sasax.cpp utils.hpp)
target_link_libraries(sasax mops)
add_test(NAME sasax COMMAND ./sasax)
