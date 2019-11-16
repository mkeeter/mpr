#include "platform.h"
#include "log.h"
#define LOG_ALIGN 10

platform_terminal_color_t log_message_color(log_type_t t) {
    switch (t) {
        case LOG_TRACE: return TERM_COLOR_BLUE;
        case LOG_INFO:  return TERM_COLOR_GREEN;
        case LOG_WARN:  return TERM_COLOR_YELLOW;
        case LOG_ERROR: return TERM_COLOR_RED;
        default:        return TERM_COLOR_WHITE;
    }
}

static struct platform_mutex_* mut = NULL;

void log_lock() {
    if (mut == NULL) {
        mut = platform_mutex_new();
    }
    platform_mutex_lock(mut);
}
void log_unlock() {
    platform_mutex_unlock(mut);
}

FILE* log_preamble(log_type_t t, const char* file, int line)
{
    static int64_t start_usec = -1;

    if (start_usec == -1) {
        start_usec = platform_get_time();
    }

    const uint64_t dt_usec = platform_get_time() - start_usec;

    FILE* out = (t == LOG_ERROR) ? stderr : stdout;

    const char* filename = platform_filename(file);

    /*  Figure out how much to pad the filename + line number */
    int pad = 0;
    for (int i=line; i; i /= 10, pad++);
    pad += strlen(filename);
    pad = LOG_ALIGN - pad;

    platform_set_terminal_color(out, log_message_color(t));
    fprintf(out, "[erizo]");

    platform_set_terminal_color(out, TERM_COLOR_WHITE);
    fprintf(out, " (%u.%06u) ", (uint32_t)(dt_usec / 1000000),
                                (uint32_t)(dt_usec % 1000000));

    platform_clear_terminal_color(out);
    fprintf(out, "%s:%i ", filename, line);

    while (pad--) {
        fputc(' ', out);
    }

    fprintf(out, "| ");
    return out;
}
