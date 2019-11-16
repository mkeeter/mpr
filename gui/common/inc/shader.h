#pragma once
#include "base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GLSL(version, shader)  "#version " #version "\n" #shader

GLuint shader_build(const GLchar* src, GLenum type);
GLuint shader_link_vgf(GLuint vs, GLuint gs, GLuint fs);
GLuint shader_link_vf(GLuint vs, GLuint fs);

#define SHADER_GET_UNIFORM_LOC(object, target) do {                     \
    object->u_##target = glGetUniformLocation(object->prog, #target);   \
    if (object->u_##target == -1) {                                     \
        log_error("Failed to get uniform " #target);                    \
    }                                                                   \
} while(0)

#ifdef __cplusplus
}
#endif
