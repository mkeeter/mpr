#include "check.hpp"

struct Image {
    Image(const uint32_t size_px)
        : size_px(size_px),
          data(CUDA_MALLOC(uint32_t, size_px * size_px))
    {
        // Nothing to do here
    }

    ~Image()
    {
        CUDA_FREE(data);
    }

    void reset() {
        cudaMemset(data, 0, size_px * size_px * sizeof(*data));
    }

    __host__ __device__
    uint32_t& operator[](size_t i) { return data[i]; }
    __host__ __device__
    uint32_t operator[](size_t i) const { return data[i]; }

    __host__ __device__
    uint32_t& operator()(uint32_t x, uint32_t y) {
        return data[x + y * size_px];
    }
    __host__ __device__
    uint32_t operator()(uint32_t x, uint32_t y) const {
        return data[x + y * size_px];
    }

    __device__ float3 voxelPos(uint3 v) const {
        return make_float3(
            2.0f * ((v.x + 0.5f) / size_px - 0.5f),
            2.0f * ((v.y + 0.5f) / size_px - 0.5f),
            2.0f * ((v.z + 0.5f) / size_px - 0.5f));
    }

    const uint32_t size_px;
    uint32_t* const __restrict__ data;
};
