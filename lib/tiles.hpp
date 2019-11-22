#include "check.hpp"

static constexpr unsigned __host__ __device__ pow(unsigned p, unsigned n)
{
    return n ? p * pow(p, n - 1) : 1;
}

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

    __device__ uint3 lowerCornerVoxel(uint32_t t) const {
        const auto i = unpack(t);
        return make_uint3(i.x * TILE_SIZE_PX,
                          i.y * TILE_SIZE_PX,
                          i.z * TILE_SIZE_PX);
    }

    __device__ uint3 unpack(uint32_t t) const {
        return make_uint3(
            t % per_side,
            (t / per_side) % per_side,
            (t / per_side) / per_side);
    }

    __device__ float3 unpackFloat(uint32_t t) const {
        const auto i = unpack(t);
        return make_float3(i.x, i.y, i.z);
    }

    __device__ float3 tileToLowerPos(uint32_t t) const {
        const auto f = unpackFloat(t);
        return make_float3(f.x / per_side,
                           f.y / per_side,
                           f.z / per_side);
    }

    __device__ float3 tileToUpperPos(uint32_t t) const {
        const auto f = unpackFloat(t);
        return make_float3((f.x + 1.0f) / per_side,
                           (f.y + 1.0f) / per_side,
                           (f.z + 1.0f) / per_side);
    }

    __device__ float3 tileToCenterPos(uint32_t t) const {
        const auto f = unpackFloat(t);
        return make_float3((f.x + 0.5f) / per_side,
                           (f.y + 0.5f) / per_side,
                           (f.z + 0.5f) / per_side);
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
