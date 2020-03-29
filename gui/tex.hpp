#pragma once
#include <cuda_gl_interop.h>
#include "context.hpp"

cudaGraphicsResource* register_texture(GLuint t);

enum Mode { RENDER_MODE_2D, RENDER_MODE_DEPTH, RENDER_MODE_NORMALS };
void copy_to_texture(const libfive::cuda::Context& ctx,
                     cudaGraphicsResource* gl_tex,
                     int texture_size_px,
                     bool append, Mode mode);
