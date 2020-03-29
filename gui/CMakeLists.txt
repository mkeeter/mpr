set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -src-in-ptx -keep --ptxas-options=-v -g -lineinfo")

add_executable(demo MACOSX_BUNDLE
    main.cpp
    interpreter.cpp
    tex.cu

    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_impl_glfw.cpp
    imgui/imgui_impl_opengl3.cpp
    imgui/imgui_widgets.cpp
    imgui-color-text-edit/TextEditor.cpp)

set_target_properties(demo PROPERTIES
    CUDA_STANDARD 11
    CXX_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(demo PRIVATE
    common/inc
    imgui
    imgui-color-text-edit)
target_link_libraries(demo glfw five-guile five-cuda)

if (APPLE)
    target_link_libraries(demo glew)

    # Hack because target_link_libraries injects -frameworks OpenGL into the CUDA
    # linker call, which fails because nvcc doesn't know about that argument
    set_property(TARGET demo APPEND_STRING PROPERTY LINK_FLAGS " -framework OpenGL")
    set_property(TARGET demo PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
else() # Ubuntu, presumably
    target_link_libraries(demo GLEW GL)
endif()
