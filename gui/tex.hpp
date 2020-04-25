/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#pragma once
#include <cuda_gl_interop.h>

// Forward declaration
namespace mpr {
struct Context;
struct Effects;
}

cudaGraphicsResource* register_texture(GLuint t);

enum Mode {
    RENDER_MODE_2D,
    RENDER_MODE_DEPTH,
    RENDER_MODE_NORMALS,
    RENDER_MODE_SSAO,
    RENDER_MODE_SHADED,
};

void copy_to_texture(const mpr::Context& ctx,
                     const mpr::Effects& effects,
                     cudaGraphicsResource* gl_tex,
                     int texture_size_px,
                     bool append, Mode mode);
