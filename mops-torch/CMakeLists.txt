cmake_minimum_required(VERSION 3.16)

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(MOPS_TORCH_MAIN_PROJECT ON)
else()
    set(MOPS_TORCH_MAIN_PROJECT OFF)
endif()

if(${MOPS_TORCH_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    # We use CACHED_LAST_CMAKE_VERSION to only print the cmake version
    # once in the configuration log
    set(CACHED_LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    message(STATUS "Running CMake version ${CMAKE_VERSION}")
endif()


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION MOPS_TORCH_VERSION)
string(STRIP ${MOPS_TORCH_VERSION} MOPS_TORCH_VERSION)
string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" MOPS_TORCH_VERSION_MAJOR "${MOPS_TORCH_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" MOPS_TORCH_VERSION_MINOR "${MOPS_TORCH_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" MOPS_TORCH_VERSION_PATCH "${MOPS_TORCH_VERSION}")

project(mops VERSION ${MOPS_TORCH_VERSION} LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(STATUS "Could not find a CUDA compiler")
endif()

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(MOPS_TORCH_MAIN_PROJECT ON)
else()
    set(MOPS_TORCH_MAIN_PROJECT OFF)
endif()

# Set a default build type if none was specified
if (${MOPS_TORCH_MAIN_PROJECT})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'relwithdebinfo' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "relwithdebinfo"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

find_package(Torch 1.11 REQUIRED)

set(BUILD_SHARED_LIBS OFF)
add_subdirectory(mops EXCLUDE_FROM_ALL)
set_target_properties(mops PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mops_torch SHARED
    "src/register.cpp"
    "src/hpe.cpp"
    "src/opsa.cpp"
    "src/sap.cpp"
    "src/opsaw.cpp"
    "src/sasaw.cpp"
    "src/utils.cpp"

    "include/mops/torch.hpp"
    "include/mops/torch/hpe.hpp"
    "include/mops/torch/opsa.hpp"
    "include/mops/torch/sap.hpp"
    "include/mops/torch/opsaw.hpp"
    "include/mops/torch/sasaw.hpp"
    "include/mops/torch/utils.hpp"
)

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(mops_torch PUBLIC MOPS_CUDA_ENABLED)
endif()

target_compile_features(mops_torch PUBLIC cxx_std_17)
target_link_libraries(mops_torch PRIVATE mops)
target_link_libraries(mops_torch PUBLIC torch)

# Create a header defining MOPS_TORCH_EXPORT for exported classes/functions
set_target_properties(mops PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
)

if (CMAKE_CUDA_COMPILER)
    set_target_properties(mops_torch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(mops_torch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()


target_include_directories(mops_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    mops_torch-config-version.cmake
    VERSION ${MOPS_TORCH_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(TARGETS mops_torch
    EXPORT mops_torch-targets
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${INCLUDE_INSTALL_DIR})

# Install files to find mops in CMake projects
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mops_torch-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mops_torch-config.cmake
    @ONLY
)
install(EXPORT mops_torch-targets
    DESTINATION ${LIB_INSTALL_DIR}/cmake/mops_torch
)
install(FILES
    ${PROJECT_BINARY_DIR}/mops_torch-config-version.cmake
    ${PROJECT_BINARY_DIR}/mops_torch-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/mops_torch
)
