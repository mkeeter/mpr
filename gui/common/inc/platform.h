#pragma once
#include "base.h"

#ifdef __cplusplus
extern "C" {
#endif

struct app_;

typedef struct platform_mmap_ platform_mmap_t;
platform_mmap_t* platform_mmap(const char* filename);
const char* platform_mmap_data(platform_mmap_t* m);
size_t platform_mmap_size(platform_mmap_t* m);
void platform_munmap(platform_mmap_t* m);

/*  Returns time in microseconds */
int64_t platform_get_time(void);

/*  Based on 8-color ANSI terminals */
typedef enum {
    TERM_COLOR_BLACK,
    TERM_COLOR_RED,
    TERM_COLOR_GREEN,
    TERM_COLOR_YELLOW,
    TERM_COLOR_BLUE,
    TERM_COLOR_MAGENTA,
    TERM_COLOR_CYAN,
    TERM_COLOR_WHITE
} platform_terminal_color_t;
void platform_set_terminal_color(FILE* f, platform_terminal_color_t c);
void platform_clear_terminal_color(FILE* f);

////////////////////////////////////////////////////////////////////////////////

/*  Threading API is a thin wrapper around pthreads */
typedef struct platform_mutex_ platform_mutex_t;
typedef struct platform_cond_ platform_cond_t;
typedef struct platform_thread_ platform_thread_t;

platform_mutex_t* platform_mutex_new();
void platform_mutex_delete(platform_mutex_t* mutex);
int platform_mutex_lock(platform_mutex_t* mutex);
int platform_mutex_unlock(platform_mutex_t* mutex);

platform_cond_t* platform_cond_new();
void platform_cond_delete(platform_cond_t* cond);
int platform_cond_wait(platform_cond_t* cond,
                       platform_mutex_t* mutex);
int platform_cond_broadcast(platform_cond_t* cond);

platform_thread_t*  platform_thread_new(void *(*run)(void *),
                                        void* data);
void platform_thread_delete(platform_thread_t* thread);
int platform_thread_join(platform_thread_t* thread);

////////////////////////////////////////////////////////////////////////////////

/*  Initializes the menu and other native features */
void platform_init(struct app_* app, int argc, char** argv);
void platform_window_bind(GLFWwindow* window);

/*  Shows a warning dialog with the given text */
void platform_warning(const char* title, const char* text);

/*  Returns the filename portion of a full path */
const char* platform_filename(const char* filepath);

#ifdef __cplusplus
}
#endif
