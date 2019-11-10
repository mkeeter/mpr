cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(libfive-cuda LANGUAGES CXX CUDA)

add_subdirectory(libfive)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -src-in-ptx -keep --ptxas-options=-v -g -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -fPIC -pedantic -Werror=switch -march=native")

add_executable(libfive-cuda-test
    libfive-cuda.cpp
    tape.cpp
    renderable.cu)
target_include_directories(libfive-cuda-test PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    libfive/libfive/include
    ${EIGEN_INCLUDE_DIRS})
target_link_libraries(libfive-cuda-test five)
set_target_properties(libfive-cuda-test PROPERTIES
    CUDA_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
  # Add the rpath to the lib
  set_property(TARGET libfive-cuda-test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
