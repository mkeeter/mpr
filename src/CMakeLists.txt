set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -src-in-ptx -keep --ptxas-options=-v -g -lineinfo")

add_library(mpr
    effects.cu
    gpu_opcode.cu
    tape.cpp
    context.cpp
    context.cu)
target_include_directories(mpr PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    libfive/libfive/include
    ${EIGEN_INCLUDE_DIRS})
target_link_libraries(mpr five)
set_target_properties(mpr PROPERTIES
    CUDA_STANDARD 11
    CXX_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON)
