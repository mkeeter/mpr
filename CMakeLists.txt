cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(libfive-cuda LANGUAGES C CXX CUDA)

# Configure everyone to use packed opcodes, which is optional
set(LIBFIVE_PACKED_OPCODES ON)
set(BUILD_STUDIO_APP OFF)
add_definitions(-DLIBFIVE_PACKED_OPCODES)

# Compile the core libfive libraries
add_subdirectory(libfive)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -g -fPIC -pedantic -Werror=switch -march=native")

option(BIG_SERVER "Running on large server" OFF)
if (${BIG_SERVER})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBIG_SERVER")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBIG_SERVER")
endif()

add_subdirectory(src)
add_subdirectory(benchmark)

if (NOT(${BIG_SERVER}))
    add_subdirectory(gui)
endif()
