add_executable(demo MACOSX_BUNDLE
    main.cpp
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_impl_glfw.cpp
    imgui/imgui_impl_opengl3.cpp
    imgui/imgui_widgets.cpp
    imgui-color-text-edit/TextEditor.cpp)

set_target_properties(demo PROPERTIES CXX_STANDARD 11)
target_include_directories(demo PRIVATE
    imgui
    imgui-color-text-edit)
target_link_libraries(demo glfw glew five-guile)

find_package(OpenGL REQUIRED)
target_link_libraries(demo ${OPENGL_LIBRARIES})
