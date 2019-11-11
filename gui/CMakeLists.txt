add_executable(demo MACOSX_BUNDLE
    main.cpp
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_impl_glfw.cpp
    imgui/imgui_impl_opengl3.cpp
    imgui/imgui_widgets.cpp)

set_target_properties(demo PROPERTIES CXX_STANDARD 11)
target_include_directories(demo PRIVATE imgui)
target_link_libraries(demo glfw)
target_link_libraries(demo glew)

find_package(OpenGL REQUIRED)
target_link_libraries(demo ${OPENGL_LIBRARIES})
