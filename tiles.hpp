#include "check.hpp"

struct Tiles {
    Tiles(const uint32_t image_size_px, const uint32_t tile_size_px)
        : per_side(image_size_px / tile_size_px),
          total(per_side * per_side),
          data(CUDA_MALLOC(uint32_t, 2 * total))
    {
        reset();
    }

    ~Tiles() {
        CHECK(cudaFree(data));
    }

    __host__ __device__ uint32_t filled(uint32_t i) const
        { return data[total * 2 - i - 1]; }
    __host__ __device__ uint32_t active(uint32_t i) const
        { return data[i * 2]; }
    __host__ __device__ uint32_t head(uint32_t i) const
        { return data[i * 2 + 1]; }

    __host__ __device__ uint32_t& filled(uint32_t i)
        { return data[total * 2 - i - 1]; }
    __host__ __device__ uint32_t& active(uint32_t i)
        { return data[i * 2]; }
    __host__ __device__ uint32_t& head(uint32_t i)
        { return data[i * 2 + 1]; }

#ifdef __CUDACC__
    __device__ void insert_filled(uint32_t index) {
        filled(atomicAdd(&num_filled, 1)) = index;
    }
    __device__ void insert_active(uint32_t index) {
        active(atomicAdd(&num_active, 1)) = index;
    }
#endif

    void reset() {
        num_active = 0;
        num_filled = 0;
        num_subtapes = 1;
    }

    const uint32_t per_side;
    const uint32_t total;

    uint32_t num_active;
    uint32_t num_filled;

    Subtapes subtapes;
    uint32_t num_subtapes;
protected:
    uint32_t* __restrict__ const data;
};
