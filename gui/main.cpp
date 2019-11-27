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
                        shapes.insert({t.id(), Renderable::build(
                                    t, 1024, 3)});
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

    std::map<libfive::Tree::Id, Renderable::Handle> shapes;
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
    ImVec2 center = ImVec2(0.0f, 0.0f);
    float scale = 100.0f; // scale = pixels per model units

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

        // Handle panning
        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseDragging()) {
                const auto drag = ImGui::GetMouseDragDelta();
                center.x += drag.x / scale;
                center.y += drag.y / scale;
                ImGui::ResetMouseDragDelta();
            }

            // Handle scrolling
            const auto scroll = io.MouseWheel;
            if (scroll) {
                // Reset accumulated scroll
                io.MouseWheel = 0.0f;

                // Start position in world coordinates
                auto mouse = io.MousePos;
                const auto sx = (mouse.x - io.DisplaySize.x / 2.0f) / scale;
                const auto sy = (mouse.y - io.DisplaySize.y / 2.0f) / scale;

                scale *= powf(1.01f, scroll);

                // End position in world coordinates
                const auto ex = (mouse.x - io.DisplaySize.x / 2.0f) / scale;
                const auto ey = (mouse.y - io.DisplaySize.y / 2.0f) / scale;

                // Shift so that world position is constant
                center.x += (ex - sx);
                center.y += (ey - sy);
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

        // Draw the shapes, and add them to the draw list
        auto background = ImGui::GetBackgroundDrawList();
        ImGui::Begin("Shapes");
            const float max_pixels = fmax(io.DisplaySize.x, io.DisplaySize.y);
            const float render_scale = max_pixels / scale / 2.0f;

            const float cx = center.x * scale + io.DisplaySize.x / 2.0f;
            const float cy = center.y * scale + io.DisplaySize.y / 2.0f;
            bool append = false;
            for (const auto& s : interpreter.shapes) {
                ImGui::Text("Shape at %p", (void*)s.first);
                ImGui::Columns(2);
                ImGui::Text("%u clauses", s.second->tape.num_clauses);
                ImGui::Text("%u registers", s.second->tape.num_regs);
                ImGui::NextColumn();
                ImGui::Text("%u constants", s.second->tape.num_constants);
                ImGui::Text("%u CSG nodes", s.second->tape.num_csg_choices);
                ImGui::Columns(1);

                {   // Timed rendering pass
                    using namespace std::chrono;
                    auto start = high_resolution_clock::now();
                        s.second->run({{center.x, -center.y}, render_scale});
                    auto end = high_resolution_clock::now();
                    auto dt = duration_cast<microseconds>(end - start);
                    ImGui::Text("Render time: %f s", dt.count() / 1e6);

                    start = high_resolution_clock::now();
                        s.second->copyToTexture(cuda_tex, TEXTURE_SIZE, append);
                    end = high_resolution_clock::now();
                    dt = duration_cast<microseconds>(end - start);
                    ImGui::Text("Texture load time: %f s", dt.count() / 1e6);
                }

                ImGui::Separator();

                background->AddImage((void*)(intptr_t)gl_tex,
                        {io.DisplaySize.x / 2.0f - max_pixels / 2.0f,
                         io.DisplaySize.y / 2.0f - max_pixels / 2.0f},
                        {io.DisplaySize.x / 2.0f + max_pixels / 2.0f,
                         io.DisplaySize.y / 2.0f + max_pixels / 2.0f});

                // Later render passes will only append to the texture,
                // instead of writing both filled and empty pixels.
                append = true;
            }

        // Draw XY axes based on current position
        background->AddLine({cx, cy}, {cx + scale, cy}, 0xFF0000FF);
        background->AddLine({cx, cy}, {cx, cy - scale}, 0xFF00FF00);

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
