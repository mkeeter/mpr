#include "object.h"
#include "rect.h"
#include "log.h"
#include "shader.h"

////////////////////////////////////////////////////////////////////////////////

static const GLchar* QUAD_VS_SRC = GLSL(330,
layout(location=0) in vec2 pos;

void main() {
    gl_Position = vec4(pos, 0.0f, 1.0f);
}
);

static const GLchar* QUAD_FS_SRC = GLSL(330,
out vec4 out_color;

void main() {
    out_color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
);

////////////////////////////////////////////////////////////////////////////////

rect_t* rect_new() {
    OBJECT_ALLOC(rect);
    glGenBuffers(1, &rect->vbo);
    glGenVertexArrays(1, &rect->vao);

    glBindBuffer(GL_ARRAY_BUFFER, rect->vbo);
    float corners[4][2] = {
        {-1.0f, -1.0f}, {1.0f, -1.0f},
        {-1.0f, 1.0f}, {1.0f, 1.0f}};
    glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
    log_gl_error();

    glBindVertexArray(rect->vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindVertexArray(0);
    log_gl_error();

    rect->vs = shader_build(QUAD_VS_SRC, GL_VERTEX_SHADER);
    rect->fs = shader_build(QUAD_FS_SRC, GL_FRAGMENT_SHADER);
    rect->prog = shader_link_vf(rect->vs, rect->fs);
    log_gl_error();

    return rect;
}

void rect_draw(rect_t* rect) {
    glBindVertexArray(rect->vao);
    glUseProgram(rect->prog);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    log_gl_error();
}
