cmake_minimum_required(VERSION 3.16)

foreach(tgt hpe)
  add_executable(${tgt} ${tgt}.cpp utils.hpp)
  target_link_libraries(${tgt} PRIVATE mops)
endforeach()
