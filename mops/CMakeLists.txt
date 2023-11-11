cmake_minimum_required(VERSION 3.16)

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION MOPS_VERSION)
string(STRIP ${MOPS_VERSION} MOPS_VERSION)
string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" MOPS_VERSION_MAJOR "${MOPS_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" MOPS_VERSION_MINOR "${MOPS_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" MOPS_VERSION_PATCH "${MOPS_VERSION}")

project(mops VERSION ${MOPS_VERSION} LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(STATUS "Could not find a CUDA compiler")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)

OPTION(MOPS_OPENMP "Try to use OpenMP when compiling Sphericart" ON)
OPTION(MOPS_ARCH_NATIVE "Try to use -march=native when compiling Sphericart" ON)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(MOPS_MAIN_PROJECT ON)
else()
    set(MOPS_MAIN_PROJECT OFF)
endif()
option(MOPS_TESTS "build and run mops unit tests" ${MOPS_MAIN_PROJECT})
option(MOPS_EXAMPLES "build and run mops examples" ${MOPS_MAIN_PROJECT})

# Set a default build type if none was specified
if (${MOPS_MAIN_PROJECT})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()


add_library(mops
    "src/capi.cpp"
    "src/opsa/opsa.cpp"
    "src/opsa/capi.cpp"
    "src/sap/sap.cpp"
    "src/sap/capi.cpp"
    "src/hpe/hpe.cpp"
    "src/hpe/capi.cpp"
    "src/opsax/opsax.cpp"
    "src/opsax/capi.cpp"
    "src/sasax/sasax.cpp"
    "src/sasax/capi.cpp"

    "include/mops.hpp"
    "include/mops.h"
    "include/mops/capi.hpp"
    "include/mops/opsa.hpp"
    "include/mops/opsa.h"
    "include/mops/sap.hpp"
    "include/mops/sap.h"
    "include/mops/hpe.hpp"
    "include/mops/hpe.h"
    "include/mops/opsax.hpp"
    "include/mops/opsax.h"
    "include/mops/sasax.hpp"
    "include/mops/sasax.h"
)

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(mops PUBLIC MOPS_CUDA_ENABLED)
    target_sources(mops
        PRIVATE
            src/opsa/opsa.cu
    )
endif()

target_compile_features(mops PUBLIC cxx_std_17)

# Create a header defining MOPS_EXPORT for exported classes/functions
set_target_properties(mops PROPERTIES
    # hide non-exported symbols by default
    CXX_VISIBILITY_PRESET hidden
)

include(GenerateExportHeader)
generate_export_header(mops
    BASE_NAME MOPS
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/mops/exports.h
)
target_compile_definitions(mops PRIVATE mops_EXPORTS)

target_include_directories(mops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


# Handle optimization and OpenMP flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wunknown-pragmas" COMPILER_SUPPORTS_WPRAGMAS)
if (MOPS_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP is enabled")
        target_link_libraries(mops PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(WARNING "Could not find OpenMP")
        if(COMPILER_SUPPORTS_WPRAGMAS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
        endif()
    endif()
else()
    if(COMPILER_SUPPORTS_WPRAGMAS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif()

if (MOPS_ARCH_NATIVE)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    # for some reason COMPILER_SUPPORTS_MARCH_NATIVE is true with Apple clang,
    # but then fails with `the clang compiler does not support '-march=native'`
    if(COMPILER_SUPPORTS_MARCH_NATIVE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "march=native is enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    else()
        message(STATUS "march=native is not supported by this compiler")
    endif()
endif()


# handle warning flags
check_cxx_compiler_flag("-Wall" COMPILER_SUPPORTS_WALL)
if(COMPILER_SUPPORTS_WALL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

check_cxx_compiler_flag("-Wextra" COMPILER_SUPPORTS_WEXTRA)
if(COMPILER_SUPPORTS_WEXTRA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()

check_cxx_compiler_flag("-Werror" COMPILER_SUPPORTS_WERROR)
if(COMPILER_SUPPORTS_WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()


if (MOPS_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (MOPS_EXAMPLES)
    add_subdirectory(examples)
endif()


#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    mops-config-version.cmake
    VERSION ${MOPS_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(TARGETS mops
    EXPORT mops-targets
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${INCLUDE_INSTALL_DIR})
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION ${INCLUDE_INSTALL_DIR})

# Install files to find mops in CMake projects
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mops-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mops-config.cmake
    @ONLY
)
install(EXPORT mops-targets
    DESTINATION ${LIB_INSTALL_DIR}/cmake/mops
)
install(FILES
    ${PROJECT_BINARY_DIR}/mops-config-version.cmake
    ${PROJECT_BINARY_DIR}/mops-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/mops
)
