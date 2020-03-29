#pragma once
#include <cuda_gl_interop.h>

// Forward declaration
namespace {
namespace cuda {
struct Context;
struct Effects;
}
}

cudaGraphicsResource* register_texture(GLuint t);

enum Mode {
    RENDER_MODE_2D,
    RENDER_MODE_DEPTH,
    RENDER_MODE_NORMALS,
    RENDER_MODE_SSAO };

void copy_to_texture(const libfive::cuda::Context& ctx,
                     const libfive::cuda::Effects& effects,
                     cudaGraphicsResource* gl_tex,
                     int texture_size_px,
                     bool append, Mode mode);
