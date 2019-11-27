add_executable(libfive-cuda-test libfive-cuda.cpp)
set_target_properties(libfive-cuda-test PROPERTIES CXX_STANDARD 11)
target_include_directories(libfive-cuda-test PUBLIC ../lib)
target_link_libraries(libfive-cuda-test five-cuda)

if(APPLE)
  # Add the rpath to the lib
  set_property(TARGET libfive-cuda-test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
