// dear imgui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// If you are new to dear imgui, see examples/README.txt and documentation at the top of imgui.cpp.
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan graphics context creation, etc.)

#include <chrono>
#include <fstream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "TextEditor.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "libfive-guile.h"
#include "renderable.hpp"

#define TEXTURE_SIZE 2048

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "glfw Error %d: %s\n", error, description);
}

////////////////////////////////////////////////////////////////////////////////

struct Range {
    int start_row;
    int end_row;
    int start_col;
    int end_col;
};

struct Interpreter {
    Interpreter() {
        // Initialize libfive-guile bindings
        scm_init_guile();
        scm_init_libfive_modules();
        scm_c_use_module("libfive kernel");

        scm_eval_sandboxed = scm_c_eval_string(R"(
(use-modules (libfive sandbox))
eval-sandboxed
)");
        scm_syntax_error_sym = scm_from_utf8_symbol("syntax-error");
        scm_numerical_overflow_sym = scm_from_utf8_symbol("numerical-overflow");
        scm_valid_sym = scm_from_utf8_symbol("valid");
        scm_result_fmt = scm_from_locale_string("~S");
        scm_other_error_fmt = scm_from_locale_string("~A: ~A");
        scm_in_function_fmt = scm_from_locale_string("In function ~A:\n~A");
        scm_syntax_error_fmt = scm_from_locale_string("~A: ~A in form ~A");
        scm_numerical_overflow_fmt = scm_from_locale_string("~A: ~A in ~A");

        // Protect all of our interpreter vars from garbage collection
        for (auto s : {scm_eval_sandboxed, scm_valid_sym,
                       scm_syntax_error_sym, scm_numerical_overflow_sym,
                       scm_result_fmt, scm_syntax_error_fmt,
                       scm_numerical_overflow_fmt, scm_other_error_fmt,
                       scm_in_function_fmt})
        {
            scm_permanent_object(s);
        }
    }

    void eval(std::string script)
    {
        auto result = scm_call_1(scm_eval_sandboxed,
                scm_from_locale_string(script.c_str()));

        //  Loop through the whole result list, looking for an invalid clause
        result_valid = true;
        for (auto r = result; !scm_is_null(r) && result_valid; r = scm_cdr(r)) {
            result_valid &= scm_is_eq(scm_caar(r), scm_valid_sym);
        }

        // If there is at least one result, then we'll convert the last one
        // into a string (with special cases for various error forms)
        auto last = scm_is_null(result) ? nullptr
                                        : scm_cdr(scm_car(scm_last_pair(result)));
        if (!result_valid) {
            /* last = '(before after key params) */
            auto before = scm_car(last);
            auto after = scm_cadr(last);
            auto key = scm_caddr(last);
            auto params = scm_cadddr(last);

            auto _stack = scm_car(scm_cddddr(last));
            SCM _str = nullptr;

            if (scm_is_eq(key, scm_syntax_error_sym)) {
                _str = scm_simple_format(SCM_BOOL_F, scm_syntax_error_fmt,
                       scm_list_3(key, scm_cadr(params), scm_cadddr(params)));
            } else if (scm_is_eq(key, scm_numerical_overflow_sym)) {
                _str = scm_simple_format(SCM_BOOL_F, scm_numerical_overflow_fmt,
                       scm_list_3(key, scm_cadr(params), scm_car(params)));
            } else {
                _str = scm_simple_format(SCM_BOOL_F, scm_other_error_fmt,
                       scm_list_2(key, scm_simple_format(
                            SCM_BOOL_F, scm_cadr(params), scm_caddr(params))));
            }
            if (!scm_is_false(scm_car(params))) {
                _str = scm_simple_format(SCM_BOOL_F, scm_in_function_fmt,
                                         scm_list_2(scm_car(params), _str));
            }
            auto str = scm_to_locale_string(_str);
            auto stack = scm_to_locale_string(_stack);

            result_err_str = std::string(str);
            result_err_stack = std::string(stack);
            result_err_range = {scm_to_int(scm_car(before)),
                                scm_to_int(scm_car(after)),
                                scm_to_int(scm_cdr(before)),
                                scm_to_int(scm_cdr(after))};

            free(str);
            free(stack);
        } else if (last) {
            char* str = nullptr;
            if (scm_to_int64(scm_length(last)) == 1) {
                auto str = scm_to_locale_string(
                        scm_simple_format(SCM_BOOL_F, scm_result_fmt,
                                          scm_list_1(scm_car(last))));

                result_str = std::string(str);
            } else {
                auto str = scm_to_locale_string(
                        scm_simple_format(SCM_BOOL_F, scm_result_fmt,
                                          scm_list_1(last)));

                result_str = std::string("(values") + str + ")";
            }
            free(str);
        } else {
            result_str = "#<eof>";
        }

        // Then iterate over the results, picking out shapes
        if (result_valid) {
            // Initialize variables and their textual positions
            std::map<libfive::Tree::Id, float> vars;
            std::map<libfive::Tree::Id, Range> var_pos;

            {   // Walk through the global variable map
                auto vs = scm_c_eval_string(R"(
                    (use-modules (libfive sandbox))
                    (hash-map->list (lambda (k v) v) vars) )");

                for (auto v = vs; !scm_is_null(v); v = scm_cdr(v))
                {
                    auto data = scm_cdar(v);
                    auto id = static_cast<libfive::Tree::Id>(
                            libfive_tree_id(scm_get_tree(scm_car(data))));
                    auto value = scm_to_double(scm_cadr(data));
                    vars[id] = value;

                    auto vp = scm_caddr(data);
                    var_pos[id] = {scm_to_int(scm_car(vp)), 0,
                                   scm_to_int(scm_cadr(vp)),
                                   scm_to_int(scm_caddr(vp))};
                }
            }

            // Then walk through the result list, picking out trees
            shapes.clear();
            while (!scm_is_null(result)) {
                for (auto r = scm_cdar(result); !scm_is_null(r); r = scm_cdr(r)) {
                    if (scm_is_shape(scm_car(r))) {
                        auto t = *scm_get_tree(scm_car(r));
                        shapes.insert({t.id(), t});
                    }
                }
                result = scm_cdr(result);
            }

            // Do something with shapes
            // Do something with vars
        }
    }

    /*  Lots of miscellaneous Scheme objects, constructed once
     *  during init() so that we don't need to build them over
     *  and over again at runtime */
    SCM scm_eval_sandboxed;
    SCM scm_port_eof_p;
    SCM scm_valid_sym;
    SCM scm_syntax_error_sym;
    SCM scm_numerical_overflow_sym;

    SCM scm_syntax_error_fmt;
    SCM scm_numerical_overflow_fmt;
    SCM scm_other_error_fmt;
    SCM scm_result_fmt;
    SCM scm_in_function_fmt;

    bool result_valid = false;
    std::string result_str;
    std::string result_err_str;
    std::string result_err_stack;
    Range result_err_range;

    std::map<libfive::Tree::Id, libfive::Tree> shapes;
};

struct Shape {
    Renderable::Handle handle;
    libfive::Tree tree;
};

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "libfive-cuda demo", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
    bool err = glewInit() != GLEW_OK;
    if (err) {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    io.Fonts->AddFontFromFileTTF("../gui/Inconsolata.ttf", 16.0f);

    // Create our text editor
    TextEditor editor;
    bool from_file = false;
    if (argc > 1) {
        std::ifstream input(argv[1]);
        if (input.is_open()) {
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(input, line)) {
                lines.emplace_back(std::move(line));
            }
            input.close();
            editor.SetTextLines(lines);
            from_file = true;
        } else {
            std::cerr << "Could not open file '" << argv[1] << "'\n";
        }
    }

    // Create the interpreter
    Interpreter interpreter;
    bool needs_eval = true;

    // Our state
    bool show_demo_window = false;
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

    // View matrix, as it were
    Eigen::Vector3f view_center{0.f, 0.0f, 0.0f};
    float view_scale = 2.0f;

    // Function to pack view_center and view_scale into the matrix
    Eigen::Affine3f model;
    Eigen::Affine3f view;
    auto update_mats = [&]() {
        model = Eigen::Affine3f::Identity();
        model.translate(view_center);
        model.scale(view_scale);

        view = Eigen::Affine3f::Identity();
        auto s = 2.0f / fmax(io.DisplaySize.x, io.DisplaySize.y);
        view.scale(Eigen::Vector3f{s, -s, 1.0f});
        view.translate(Eigen::Vector3f{-io.DisplaySize.x / 2.0f, -io.DisplaySize.y / 2.0f, 0.0f});
    };
    update_mats();

    std::map<libfive::Tree::Id, Shape> shapes;

    // Generate a texture which we'll draw into
    GLuint gl_tex;
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 TEXTURE_SIZE,
                 TEXTURE_SIZE, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    auto cuda_tex = Renderable::registerTexture(gl_tex);

    bool just_saved = false;

    // Main loop
    int render_size = 1024;
    int render_dimension = 2;
    int render_mode = 0;

    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwWaitEventsTimeout(0.1f);

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Rebuild the transform matrix, in case the window size has changed
        update_mats();

        // Handle panning
        if (!io.WantCaptureMouse) {
            // Start position in world coordinates
            const Eigen::Vector3f mouse = Eigen::Vector3f{
                io.MousePos.x, io.MousePos.y, 0.0f};

            if (ImGui::IsMouseDragging()) {
                const auto d = ImGui::GetMouseDragDelta();
                const Eigen::Vector3f drag(d.x, d.y, 0.0f);
                view_center += (model * view * (mouse - drag)) -
                               (model * view * mouse);
                update_mats();
                ImGui::ResetMouseDragDelta();
            }

            // Handle scrolling
            const auto scroll = io.MouseWheel;
            if (scroll) {
                // Reset accumulated scroll
                io.MouseWheel = 0.0f;

                // Start position in world coordinates
                const Eigen::Vector3f start = (model * view * mouse);

                // Update matrix
                view_scale *= powf(1.01f, scroll);
                update_mats();

                const Eigen::Vector3f end = (model * view * mouse);

                // Shift so that world position is constant
                view_center -= (end - start);
                update_mats();
            }
        }

        if (!io.WantCaptureKeyboard) {
            if (io.KeySuper && io.KeysDown[GLFW_KEY_S]) {
                if (!just_saved && from_file) {
                    std::ofstream output(argv[1]);
                    if (output.is_open()) {
                        for (auto& line: editor.GetTextLines()) {
                            output << line << "\n";
                        }
                        output.close();
                    } else {
                        std::cerr << "Failed to save to '" << argv[1] << "'\n";
                    }
                    just_saved = true;
                }
            } else {
                just_saved = false;
            }
        }

        // Draw main menu
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("View")) {
                ImGui::Checkbox("Show demo window", &show_demo_window);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        if (show_demo_window) {
            ImGui::ShowDemoWindow(&show_demo_window);
        }

        // Draw the interpreter window and handle re-evaluation as needed
        ImGui::Begin("Text editor");
            if (needs_eval) {
                interpreter.eval(editor.GetText());

                // Erase shapes that are no longer in the script
                auto itr = shapes.begin();
                while (itr != shapes.end()) {
                    if (interpreter.shapes.find(itr->first) == interpreter.shapes.end()) {
                        itr = shapes.erase(itr);
                    } else {
                        ++itr;
                    }
                }
                // Create new shapes from the script
                for (auto& t : interpreter.shapes) {
                    if (shapes.find(t.first) == shapes.end()) {
                        Shape s = { Renderable::build(
                                        t.second,
                                        render_size,
                                        render_dimension),
                                    t.second };
                        shapes.emplace(t.first, std::move(s));
                    }
                }
            }

            float size = ImGui::GetContentRegionAvail().y;
            if (interpreter.result_valid) {
                size -= ImGui::GetFrameHeight() *
                        (std::count(interpreter.result_str.begin(),
                                    interpreter.result_str.end(), '\n') + 1);
            } else {
                size -= ImGui::GetFrameHeight() *
                        (std::count(interpreter.result_err_str.begin(),
                                    interpreter.result_err_str.end(), '\n') + 1);
            }

            needs_eval = editor.Render("TextEditor", ImVec2(0, size));
            if (interpreter.result_valid) {
                ImGui::Text("%s", interpreter.result_str.c_str());
            } else {
                ImGui::Text("%s", interpreter.result_err_str.c_str());
            }
        ImGui::End();

        ImGui::Begin("Settings");
            ImGui::Text("Render size:");
            ImGui::RadioButton("512", &render_size, 512);
            ImGui::SameLine();
            ImGui::RadioButton("1024", &render_size, 1024);
            ImGui::SameLine();
            ImGui::RadioButton("2048", &render_size, 2048);

            ImGui::Text("Dimension:");
            ImGui::RadioButton("2D", &render_dimension, 2);
            ImGui::SameLine();
            ImGui::RadioButton("3D", &render_dimension, 3);

            if (render_dimension == 3) {
                ImGui::Text("Render mode:");
                ImGui::RadioButton("Heightmap", &render_mode, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Normals", &render_mode, 1);
            }
        ImGui::End();

        // Draw the shapes, and add them to the draw list
        auto background = ImGui::GetBackgroundDrawList();

        ImGui::Begin("Shapes");
            bool append = false;

            for (auto& s : shapes) {
                ImGui::Text("Shape at %p", (void*)s.first);
                ImGui::Columns(2);
                ImGui::Text("%u clauses", s.second.handle->tape.num_clauses);
                ImGui::Text("%u registers", s.second.handle->tape.num_regs);
                ImGui::NextColumn();
                ImGui::Text("%u constants", s.second.handle->tape.num_constants);
                ImGui::Text("%u CSG nodes", s.second.handle->tape.num_csg_choices);
                ImGui::Columns(1);

                {   // Timed rendering pass
                    using namespace std::chrono;
                    auto start = high_resolution_clock::now();
                        s.second.handle->run({model.matrix()});
                    auto end = high_resolution_clock::now();
                    auto dt = duration_cast<microseconds>(end - start);
                    ImGui::Text("Render time: %f s", dt.count() / 1e6);

                    start = high_resolution_clock::now();
                        s.second.handle->copyToTexture(cuda_tex, TEXTURE_SIZE,
                                                       append, render_mode);
                    end = high_resolution_clock::now();
                    dt = duration_cast<microseconds>(end - start);
                    ImGui::Text("Texture load time: %f s", dt.count() / 1e6);
                }

                if (render_size != (int)s.second.handle->image.size_px ||
                    render_dimension != (int)s.second.handle->dimension())
                {
                    s.second.handle.reset();
                    auto h = Renderable::build(s.second.tree,
                                               render_size,
                                               render_dimension);
                    s.second.handle = std::move(h);
                }
                if (ImGui::Button("Save shape.frep")) {
                    auto a = libfive::Archive();
                    a.addShape(s.second.tree);
                    std::ofstream out("shape.frep");
                    if (out.is_open()) {
                        a.serialize(out);
                    } else {
                        std::cerr << "Could not open shape.frep\n";
                    }
                }

                ImGui::Separator();

                // Later render passes will only append to the texture,
                // instead of writing both filled and empty pixels.
                append = true;
            }

            const float max_pixels = fmax(io.DisplaySize.x, io.DisplaySize.y);
            background->AddImage((void*)(intptr_t)gl_tex,
                    {io.DisplaySize.x / 2.0f - max_pixels / 2.0f,
                     io.DisplaySize.y / 2.0f - max_pixels / 2.0f},
                    {io.DisplaySize.x / 2.0f + max_pixels / 2.0f,
                     io.DisplaySize.y / 2.0f + max_pixels / 2.0f});


            // Draw XY axes based on current position
            {
                Eigen::Vector3f center = Eigen::Vector3f::Zero();
                Eigen::Vector3f ax = Eigen::Vector3f{1.0f, 0.0f, 0.0f};
                Eigen::Vector3f ay = Eigen::Vector3f{0.0f, 1.0f, 0.0f};
                Eigen::Vector3f az = Eigen::Vector3f{0.0f, 0.0f, 1.0f};

                for (auto pt : {&center, &ax, &ay, &az}) {
                    *pt = (model * view).inverse() * (*pt);
                }
                background->AddLine({center.x(), center.y()},
                                    {ax.x(), ax.y()}, 0xFF0000FF);
                background->AddLine({center.x(), center.y()},
                                    {ay.x(), ay.y()}, 0xFF00FF00);
                background->AddLine({center.x(), center.y()},
                                    {az.x(), az.y()}, 0xFFFF0000);
            }

        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
