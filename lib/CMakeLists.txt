set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -src-in-ptx -keep --ptxas-options=-v -g -lineinfo")

add_library(five-cuda
    tape.cpp
    renderable.cu)
set_target_properties(five-cuda PROPERTIES CXX_STANDARD 11)

target_include_directories(five-cuda PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}
    libfive/libfive/include
    ${EIGEN_INCLUDE_DIRS})
target_link_libraries(five-cuda five)
set_target_properties(five-cuda PROPERTIES
    CUDA_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON)
