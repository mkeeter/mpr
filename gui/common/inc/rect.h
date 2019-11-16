#pragma once
#include "base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rect_ {
    GLuint vao;
    GLuint vbo;

    GLuint vs;
    GLuint fs;
    GLuint prog;
} rect_t;

rect_t* rect_new();
void rect_draw(rect_t* q);

#ifdef __cplusplus
}
#endif
