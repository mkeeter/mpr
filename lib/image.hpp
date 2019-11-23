#include "check.hpp"

struct Image {
    Image(const uint32_t size_px)
        : size_px(size_px),
          data(CUDA_MALLOC(uint8_t, size_px * size_px))
    {
        // Nothing to do here
    }

    ~Image()
    {
        CUDA_CHECK(cudaFree(data));
    }

    void reset() {
        cudaMemset(data, 0, size_px * size_px);
    }

    __host__ __device__
    uint8_t& operator[](size_t i) { return data[i]; }
    __host__ __device__
    uint8_t operator[](size_t i) const { return data[i]; }

    __host__ __device__
    uint8_t& operator()(uint32_t x, uint32_t y) {
        return data[x + y * size_px];
    }

    const uint32_t size_px;
    uint8_t* const __restrict__ data;
};
