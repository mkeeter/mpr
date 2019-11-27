cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(libfive-cuda LANGUAGES C CXX CUDA)

# Configure everyone to use packed opcodes, which is optional
set(LIBFIVE_PACKED_OPCODES ON)
set(BUILD_STUDIO_APP OFF)
add_definitions(-DLIBFIVE_PACKED_OPCODES)

# Compile the core libfive libraries
add_subdirectory(libfive)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -fPIC -pedantic -Werror=switch -march=native")

add_subdirectory(lib)
add_subdirectory(gui)
add_subdirectory(benchmark)
