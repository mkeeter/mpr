#include "context.hpp"
#include "effects.hpp"

#include "tex.hpp"

cudaGraphicsResource* register_texture(GLuint t)
{
    cudaGraphicsResource* gl_tex;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&gl_tex, t, GL_TEXTURE_2D,
                                      cudaGraphicsMapFlagsWriteDiscard));
    return gl_tex;
}

////////////////////////////////////////////////////////////////////////////////

__global__
void copy_2d_to_surface(int32_t* const __restrict__ image,
                        int image_size_px,
                        cudaSurfaceObject_t surf,
                        int texture_size_px, bool append)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size_px && y < texture_size_px) {
        const uint32_t px = x * image_size_px / texture_size_px;
        const uint32_t py = y * image_size_px / texture_size_px;
        const auto h = image[px + py * image_size_px];
        if (h) {
            surf2Dwrite(0xFFFFFFFF, surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__global__
void copy_depth_to_surface(int32_t* const __restrict__ image,
                        int image_size_px,
                        cudaSurfaceObject_t surf,
                        int texture_size_px, bool append)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size_px && y < texture_size_px) {
        const uint32_t px = x * image_size_px / texture_size_px;
        const uint32_t py = y * image_size_px / texture_size_px;
        auto h = image[px + py * image_size_px];
        if (h) {
            h = (h * 255) / image_size_px;
            surf2Dwrite(0x00FFFFFF | (h << 24),
                        surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__global__
void copy_normals_to_surface(int32_t* const __restrict__ image,
                             uint32_t* const __restrict__ normals,
                             int image_size_px,
                             cudaSurfaceObject_t surf,
                             int texture_size_px, bool append)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size_px && y < texture_size_px) {
        const uint32_t px = x * image_size_px / texture_size_px;
        const uint32_t py = y * image_size_px / texture_size_px;
        const auto h = image[px + py * image_size_px];
        if (h) {
            surf2Dwrite(normals[px + py * image_size_px],
                        surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

__global__
void copy_ssao_to_surface(int32_t* const __restrict__ image,
                          int32_t* const __restrict__ ssao,
                          int image_size_px,
                          cudaSurfaceObject_t surf,
                          int texture_size_px, bool append)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < texture_size_px && y < texture_size_px) {
        const uint32_t px = x * image_size_px / texture_size_px;
        const uint32_t py = y * image_size_px / texture_size_px;
        const auto h = image[px + py * image_size_px];
        if (h) {
            auto s = ssao[px + py * image_size_px];
            surf2Dwrite(0xFF000000 | s | (s << 8) | (s << 16),
                        surf, x*4, y);
        } else if (!append) {
            surf2Dwrite(0, surf, x*4, y);
        }
    }
}

void copy_to_texture(const libfive::cuda::Context& ctx,
                     const libfive::cuda::Effects& effects,
                     cudaGraphicsResource* gl_tex,
                     int texture_size_px,
                     bool append,
                     Mode mode)
{
    cudaArray* array;
    CUDA_CHECK(cudaGraphicsMapResources(1, &gl_tex));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, gl_tex, 0, 0));

    // Specify texture
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    // Surface object??!
    cudaSurfaceObject_t surf = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res_desc));
    CUDA_CHECK(cudaDeviceSynchronize());

    const unsigned u = (texture_size_px + 15) / 16;
    switch (mode) {
        case RENDER_MODE_2D:
            copy_2d_to_surface<<<dim3(u, u), dim3(16, 16)>>>(
                    ctx.stages[3].filled.get(),
                    ctx.image_size_px,
                    surf, texture_size_px, append);
            break;
        case RENDER_MODE_DEPTH:
            copy_depth_to_surface<<<dim3(u, u), dim3(16, 16)>>>(
                    ctx.stages[3].filled.get(),
                    ctx.image_size_px,
                    surf, texture_size_px, append);
            break;
        case RENDER_MODE_NORMALS:
            copy_normals_to_surface<<<dim3(u, u), dim3(16, 16)>>>(
                    ctx.stages[3].filled.get(),
                    ctx.normals.get(),
                    ctx.image_size_px,
                    surf, texture_size_px, append);
            break;
        case RENDER_MODE_SSAO:
            copy_ssao_to_surface<<<dim3(u, u), dim3(16, 16)>>>(
                    ctx.stages[3].filled.get(),
                    effects.image.get(),
                    ctx.image_size_px,
                    surf, texture_size_px, append);
            break;
        default: break;
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDestroySurfaceObject(surf));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &gl_tex));
}
