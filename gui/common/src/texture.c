#include "texture.h"
#include "log.h"
#include "object.h"

texture_t* texture_new(GLsizei width, GLsizei height) {
    OBJECT_ALLOC(texture);
    glGenTextures(1, &texture->tex);

    glBindTexture(GL_TEXTURE_2D, texture->tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    log_gl_error();

    texture->buf = calloc(width * height, sizeof(uint8_t) * 4);
    texture->width = width;
    texture->height = height;

    return texture;
}

void texture_load_mono(texture_t* texture, const uint8_t* data) {
    uint8_t* c = texture->buf;
    for (unsigned y=0; y < texture->height; ++y) {
        const uint8_t* d = data + (texture->height - y - 1) * texture->width;
        for (unsigned x=0; x < texture->height; ++x) {
            *c++ = 255; // R
            *c++ = 255; // G
            *c++ = 255; // B
            *c++ = *d++;
        }
    }
    glBindTexture(GL_TEXTURE_2D, texture->tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            texture->width, texture->height,
             GL_RGBA, GL_UNSIGNED_BYTE,
             texture->buf);
    log_gl_error();
}
