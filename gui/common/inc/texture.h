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
void texture_load_mono(texture_t* texture, const uint8_t* data);

#ifdef __cplusplus
}
#endif
