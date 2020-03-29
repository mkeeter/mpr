#include "effects.hpp"

namespace libfive {
namespace cuda {

__global__
void draw_ssao(const int32_t* const __restrict__ depth,
               const uint32_t* const __restrict__ norm,

               const Eigen::Matrix<float, 64, 3> ssao_kernel,
               const Eigen::Matrix<float, 16*16, 3> ssao_rvecs,
               const int image_size_px,

               int32_t* const __restrict__ output)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    constexpr float RADIUS = 0.1f;

    if (x >= image_size_px && y >= image_size_px) {
        return;
    }

    const int h = depth[x + y * image_size_px];
    if (!h) {
        return;
    }

    const float3 pos = make_float3(
        2.0f * ((x + 0.5f) / image_size_px - 0.5f),
        2.0f * ((y + 0.5f) / image_size_px - 0.5f),
        2.0f * ((h + 0.5f) / image_size_px - 0.5f));

    // Based on http://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
    const uint32_t n = norm[x + y * image_size_px];

    // Get normal from image
    const float dx = (float)(n & 0xFF) - 128.0f;
    const float dy = (float)((n >> 8) & 0xFF) - 128.0f;
    const float dz = (float)((n >> 16) & 0xFF) - 128.0f;
    Eigen::Vector3f normal = Eigen::Vector3f{dx, dy, dz}.normalized();

    Eigen::Vector3f rvec = ssao_rvecs.row((threadIdx.x % 16) * 16 + (threadIdx.y % 16));
    Eigen::Vector3f tangent = (rvec - normal * rvec.dot(normal)).normalized();
    Eigen::Vector3f bitangent = normal.cross(tangent);
    Eigen::Matrix3f tbn;
    tbn.col(0) = tangent;
    tbn.col(1) = bitangent;
    tbn.col(2) = normal;

    float occlusion = 0.0f;
    for (unsigned i=0; i < ssao_kernel.rows(); ++i) {
        Eigen::Vector3f sample_pos =
            tbn * ssao_kernel.row(i).transpose() * RADIUS +
            Eigen::Vector3f{pos.x, pos.y, pos.z};

        const unsigned px = (sample_pos.x() / 2.0f + 0.5f) * image_size_px;
        const unsigned py = (sample_pos.y() / 2.0f + 0.5f) * image_size_px;
        const unsigned actual_h =
            (px < image_size_px && py < image_size_px)
            ? depth[px + py * image_size_px]
            : 0;
        const float actual_z = 2.0f * ((actual_h + 0.5f) / image_size_px - 0.5f);

        const auto dz = fabsf(sample_pos.z() - actual_z);
        if (dz < RADIUS) {
            occlusion += sample_pos.z() <= actual_z;
        } else if (dz < RADIUS * 2.0f) {
            if (sample_pos.z() <= actual_z) {
                occlusion += powf((RADIUS - (dz - RADIUS)) / RADIUS, 2.0f);
            }
        }
    }
    occlusion = 1.0 - (occlusion / ssao_kernel.rows());
    const uint8_t o = occlusion * 255;
    output[x + y * image_size_px] = o;
}

////////////////////////////////////////////////////////////////////////////////

__global__
void blur_ssao(const int32_t* const __restrict__ image,
               const int32_t* const __restrict__ ssao,
               const int image_size_px,
               int32_t* const __restrict__ output)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= image_size_px || y >= image_size_px) {
        return;
    }

    const int BLUR_RADIUS = 2;

    float best = 1000000.0f;
    float value = 0.0f;
    auto run = [=, &best, &value](int xmin, int ymin) {
        float sum = 0.0f;
        float count = 0.0f;
        for (int i=0; i <= BLUR_RADIUS; ++i) {
            for (int j=0; j <= BLUR_RADIUS; ++j) {
                const int tx = x + xmin + i;
                const int ty = y + ymin + j;
                if (tx >= 0 && tx < image_size_px &&
                    ty >= 0 && ty < image_size_px)
                {
                    if (image[tx + ty * image_size_px]) {
                        sum += ssao[tx + ty * image_size_px];
                        count++;
                    }
                }
            }
        }
        const float mean = sum / count;
        float stdev = 0.0f;
        for (int i=0; i <= BLUR_RADIUS; ++i) {
            for (int j=0; j <= BLUR_RADIUS; ++j) {
                const int tx = xmin + i;
                const int ty = ymin + j;
                if (tx >= 0 && tx < image_size_px &&
                    ty >= 0 && ty < image_size_px)
                {
                    if (image[tx + ty * image_size_px]) {
                        const float d = (mean - ssao[tx + ty * image_size_px]);
                        stdev += d * d;
                    }
                }
            }
        }
        stdev /= count - 1.0f;
        stdev = sqrtf(stdev);
        if (stdev < best) {
            best = stdev;
            value = mean;
        }
    };

    for (unsigned i=0; i < 4; ++i) {
        run((i & 1) ? 0 : -BLUR_RADIUS,
            (i & 2) ? 0 : -BLUR_RADIUS);
    }
    output[x + y * image_size_px] = value;
}

////////////////////////////////////////////////////////////////////////////////

Effects::Effects(int32_t image_size_px)
    : image_size_px(image_size_px),
      tmp(CUDA_MALLOC(int32_t, pow(image_size_px, 2))),
      image(CUDA_MALLOC(int32_t, pow(image_size_px, 2)))
{
    // Based on http://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
    for (unsigned i = 0; i < ssao_kernel.rows(); ++i) {
        ssao_kernel.row(i) = Eigen::RowVector3f{
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            (float)(rand()) / (float)(RAND_MAX) };
        ssao_kernel.row(i) /= ssao_kernel.row(i).norm();

        // Scale to keep most samples near the center
        float scale = float(i) / float(ssao_kernel.rows() - 1);
        scale = (scale * scale) * 0.9f + 0.1f;
        ssao_kernel.row(i) *= scale;
    }
    for (unsigned i = 0; i < ssao_rvecs.rows(); ++i) {
        ssao_rvecs.row(i) = Eigen::RowVector3f{
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            2.0f * ((float)(rand()) / (float)(RAND_MAX) - 0.5f),
            0.0f };
        ssao_rvecs.row(i) /= ssao_rvecs.row(i).norm();
    }
}

void Effects::drawSSAO(const int32_t* depth,
                       const uint32_t* norm)
{
    CUDA_CHECK(cudaMemset(image.get(), 0, sizeof(int32_t) * pow(image_size_px, 2)));
    const unsigned u = (image_size_px + 15) / 16;
    draw_ssao<<<dim3(u, u), dim3(16, 16)>>>(
            depth, norm,
            ssao_kernel, ssao_rvecs, image_size_px,
            tmp.get());
    blur_ssao<<<dim3(u, u), dim3(16, 16)>>>(
            depth, tmp.get(), image_size_px, image.get());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}   // namespace cuda
}   // namespace libfive
