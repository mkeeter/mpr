function(benchmark)
    set(files "")
    foreach(f ${ARGN})
        list(APPEND files "${f}.cpp")
    endforeach(f)
    add_executable(${ARGV0} ${files})
    set_target_properties(${ARGV0} PROPERTIES CXX_STANDARD 11)
    target_include_directories(${ARGV0} PUBLIC ../lib)
    target_link_libraries(${ARGV0} five-cuda)

    if(APPLE)
      # Add the rpath to the lib
      set_property(TARGET ${ARGV0} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endfunction()

benchmark(render_2d_table stats)
benchmark(render_3d_table stats)
benchmark(brute stats)

benchmark(render_2d)
benchmark(render_3d)
benchmark(circle)
benchmark(donut)
benchmark(dump_tape)
benchmark(tape_shortening)
benchmark(ssao)
benchmark(tape_building_time)

################################################################################

add_executable(tape_size tape_size.cpp ../gui/interpreter.cpp)
set_target_properties(tape_size PROPERTIES CXX_STANDARD 11)
target_include_directories(tape_size PUBLIC ../lib)
target_link_libraries(tape_size five-cuda five-guile)

if(APPLE)
  # Add the rpath to the lib
  set_property(TARGET tape_size PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
