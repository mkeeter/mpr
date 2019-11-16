#include "log.h"
#include "shader.h"

GLuint shader_build(const GLchar* src, GLenum type) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);

        GLchar* buf = (GLchar*)malloc(len + 1);
        glGetShaderInfoLog(shader, len, NULL, buf);
        log_error_and_abort("Failed to build shader: %s", buf);
        free(buf);
    }

    return shader;
}

void shader_check_link(GLuint program) {
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);

        GLchar* buf = (GLchar*)malloc(len + 1);
        glGetProgramInfoLog(program, len, NULL, buf);
        log_error_and_abort("Failed to link program: %s", buf);
    }
}

GLuint shader_link_vf(GLuint vs, GLuint fs) {
    GLuint program = glCreateProgram();

    glAttachShader(program, vs);
    glAttachShader(program, fs);

    glLinkProgram(program);
    shader_check_link(program);

    return program;
}

GLuint shader_link_vgf(GLuint vs, GLuint gs, GLuint fs) {
    GLuint program = glCreateProgram();

    glAttachShader(program, vs);
    glAttachShader(program, gs);
    glAttachShader(program, fs);

    glLinkProgram(program);
    shader_check_link(program);

    return program;
}
