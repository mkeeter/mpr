#include "check.hpp"

struct Image {
    Image(const uint32_t size_px)
        : size_px(size_px),
          data(CUDA_MALLOC(uint16_t, size_px * size_px))
    {
        // Nothing to do here
    }

    ~Image()
    {
        CUDA_CHECK(cudaFree(data));
    }

    void reset() {
        cudaMemset(data, 0, size_px * size_px * sizeof(*data));
    }

    __host__ __device__
    uint16_t& operator[](size_t i) { return data[i]; }
    __host__ __device__
    uint16_t operator[](size_t i) const { return data[i]; }

    __host__ __device__
    uint16_t& operator()(uint32_t x, uint32_t y) {
        return data[x + y * size_px];
    }
    __host__ __device__
    uint16_t operator()(uint32_t x, uint32_t y) const {
        return data[x + y * size_px];
    }

    const uint32_t size_px;
    uint16_t* const __restrict__ data;
};
