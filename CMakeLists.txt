cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(libfive-cuda LANGUAGES C CXX CUDA)

add_subdirectory(libfive)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -fPIC -pedantic -Werror=switch -march=native")
add_subdirectory(lib)

add_executable(libfive-cuda-test libfive-cuda.cpp)
set_target_properties(libfive-cuda-test PROPERTIES CXX_STANDARD 11)
target_include_directories(libfive-cuda-test PUBLIC lib)
target_link_libraries(libfive-cuda-test five-cuda)

add_subdirectory(gui)

if(APPLE)
  # Add the rpath to the lib
  set_property(TARGET libfive-cuda-test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
