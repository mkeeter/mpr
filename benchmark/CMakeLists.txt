macro(benchmark)
    add_executable(${ARGV0} ${ARGV0}.cpp)
    set_target_properties(${ARGV0} PROPERTIES CXX_STANDARD 11)
    target_include_directories(${ARGV0} PUBLIC ../lib)
    target_link_libraries(${ARGV0} five-cuda)

    if(APPLE)
      # Add the rpath to the lib
      set_property(TARGET ${ARGV0} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endmacro()

benchmark(render_2d)
benchmark(render_3d)
