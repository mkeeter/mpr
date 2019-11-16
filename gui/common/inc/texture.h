#pragma once
#include "base.h"

#ifdef __cplusplus
extern "C" {
#endif

// All textures are RGBA
typedef struct texture_ {
    uint8_t* buf;
    GLsizei width;
    GLsizei height;
    GLuint tex;
} texture_t;

texture_t* texture_new(GLsizei width, GLsizei height);

#ifdef __cplusplus
}
#endif
