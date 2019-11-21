#include "check.hpp"
#include "ipow.hpp"

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
struct Tiles {
    Tiles(const uint32_t image_size_px)
        : per_side(image_size_px / TILE_SIZE_PX),
          total(pow(per_side, DIMENSION)),
          data(CUDA_MALLOC(uint32_t, 2 * total))
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

    __host__ __device__ uint32_t& filled(uint32_t i)
        { return data[total - i - 1]; }
    __host__ __device__ uint32_t& active(uint32_t i)
        { return data[i]; }
    __host__ __device__ uint32_t& head(uint32_t t)
        { return data[total + t]; }

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
