#pragma once
#include "base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LOG_TRACE,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
} log_type_t;

void log_lock();
void log_unlock();
FILE* log_preamble(log_type_t t, const char* file, int line);

#define log_print(t, ...) do {                          \
    log_lock();                                         \
    FILE* out = log_preamble(t, __FILE__, __LINE__);    \
    fprintf(out, __VA_ARGS__);                          \
    fprintf(out, "\n");                                 \
    log_unlock();                                       \
} while (0)

#define log_trace(...)  log_print(LOG_TRACE, __VA_ARGS__)
#define log_info(...)   log_print(LOG_INFO,  __VA_ARGS__)
#define log_warn(...)   log_print(LOG_WARN,  __VA_ARGS__)
#define log_error(...)  log_print(LOG_ERROR, __VA_ARGS__)
#define log_error_and_abort(...) do {                           \
    log_print(LOG_ERROR, __VA_ARGS__);                          \
    exit(-1);                                                   \
} while (0)


#define LOG_GL_ERR_CASE(s) case s: err = #s; break
#define log_gl_error() do {                                         \
    GLenum status = glGetError();                                   \
    if (status != GL_NO_ERROR) {                                    \
        const char* err = NULL;                                     \
        switch (status) {                                           \
            LOG_GL_ERR_CASE(GL_NO_ERROR);                           \
            LOG_GL_ERR_CASE(GL_INVALID_ENUM);                       \
            LOG_GL_ERR_CASE(GL_INVALID_VALUE);                      \
            LOG_GL_ERR_CASE(GL_INVALID_OPERATION);                  \
            LOG_GL_ERR_CASE(GL_INVALID_FRAMEBUFFER_OPERATION);      \
            LOG_GL_ERR_CASE(GL_OUT_OF_MEMORY);                      \
            default: break;                                         \
        }                                                           \
        if (err) {                                                  \
            log_error("OpenGL error: %s (0x%x)", err, status);      \
        } else {                                                    \
            log_error("Unknown OpenGL error: 0x%x", status);        \
        }                                                           \
    }                                                               \
} while(0)

#ifdef __cplusplus
}
#endif
