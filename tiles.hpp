#include "check.hpp"

struct Tiles {
    Tiles(const uint32_t image_size_px, const uint32_t tile_size_px)
        : per_side(image_size_px / tile_size_px),
          total(per_side * per_side),
          data(CUDA_MALLOC(uint32_t, 3 * total))
    {
        reset();
    }

    ~Tiles() {
        CHECK(cudaFree(data));
    }

    __host__ __device__ uint32_t filled(uint32_t i) const
        { return data[total - i - 1]; }
    __host__ __device__ uint32_t active(uint32_t i) const
        { return data[i]; }
    __host__ __device__ uint32_t head(uint32_t t) const
        { return data[total + t]; }
    __host__ __device__ uint32_t choices(uint32_t t) const
        { return data[2*total + t]; }

    __host__ __device__ uint32_t& filled(uint32_t i)
        { return data[total - i - 1]; }
    __host__ __device__ uint32_t& active(uint32_t i)
        { return data[i]; }
    __host__ __device__ uint32_t& head(uint32_t t)
        { return data[total + t]; }
    __host__ __device__ uint32_t& choices(uint32_t t)
        { return data[2*total + t]; }

#ifdef __CUDACC__
    __device__ void insert_filled(uint32_t t) {
        filled(atomicAdd(&num_filled, 1)) = t;
    }
    __device__ void insert_active(uint32_t t) {
        active(atomicAdd(&num_active, 1)) = t;
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
