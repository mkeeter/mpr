function(benchmark)
    set(files "")
    get_filename_component(TARGET ${ARGV0} NAME_WE)
    add_executable(${TARGET} ${ARGN})
    set_target_properties(${TARGET} PROPERTIES
        CUDA_STANDARD 11
        CXX_STANDARD 11
        CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${TARGET} PUBLIC ../lib)
    target_link_libraries(${TARGET} mpr)

    if(APPLE)
      # Add the rpath to the lib
      set_property(TARGET ${TARGET} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endfunction()

benchmark(render_2d_table.cpp stats.cpp)
benchmark(render_3d_table.cpp stats.cpp)
benchmark(brute.cu stats.cpp)

benchmark(render_2d.cpp)
benchmark(render_3d.cpp)
benchmark(render_3d_anim.cpp)
benchmark(render_2d_heatmap.cpp)
benchmark(render_3d_heatmap.cpp)
benchmark(render_effects.cpp)

benchmark(csg_anim.cpp ../libfive/libfive/test/util/shapes.cpp)
benchmark(circle.cpp)
benchmark(print_tape_table.cpp)
benchmark(dump_tape.cpp)
benchmark(tape_shortening.cpp)
benchmark(tape_building_time.cpp)
