cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cmake_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -g -lineinfo")

add_executable(cmake_cuda libfive-cuda.cu tape.cpp)
target_include_directories(cmake_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(cmake_cuda PRIVATE /Developer/NVIDIA/CUDA-10.1/samples/common/inc)
target_include_directories(cmake_cuda PRIVATE libfive/libfive/include)
target_link_libraries(cmake_cuda ${CMAKE_SOURCE_DIR}/libfive/build/libfive/src/libfive.dylib)
set_property(TARGET cmake_cuda PROPERTY CUDA_STANDARD 11)
set_target_properties(cmake_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
  # Add the rpath to the lib
  set_property(TARGET cmake_cuda PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
