/*
Simple GUI to demonstrate the reference implementation of
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
