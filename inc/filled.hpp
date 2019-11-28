#include "check.hpp"

template <unsigned TILE_SIZE_PX>
struct Filled
{
    Filled(const uint32_t image_size_px)
        : per_side(image_size_px / TILE_SIZE_PX),
          total(pow(per_side, 2)),
          data(CUDA_MALLOC(uint32_t, total))
    {
        reset();
    }

    void reset() {
        memset(data, 0, total * sizeof(*data));
    }

    __host__ __device__ uint3 unpack(uint32_t t) const {
        return make_uint3(
            t % per_side,
            (t / per_side) % per_side,
            (t / per_side) / per_side);
    }

    // Returns the z index of the filled tile at t, if present
    //
    // This ignores Z coordinates, because filled tiles occlude
    // anything behind them, so only the highest Z value matters.
    __device__ uint32_t& filled(uint32_t t) {
        uint3 i = unpack(t);
        return data[i.x + i.y * per_side];
    }
    __device__ uint32_t filled(uint32_t t) const {
        uint3 i = unpack(t);
        return data[i.x + i.y * per_side];
    }

    // Returns the Z height of the given pixels
    __device__ uint32_t at(uint32_t px, uint32_t py) const {
        const auto tx = px / TILE_SIZE_PX;
        const auto ty = py / TILE_SIZE_PX;
        return data[tx + ty * per_side];
    }

#ifdef __CUDACC__
    // Marks that the tile at t is filled.  Filled tiles occlude
    // everything behind them, so this is an atomicMax operation.
    __device__ void insert(uint32_t t) {
        uint3 i = unpack(t);
        atomicMax(&filled(t), i.z * TILE_SIZE_PX + TILE_SIZE_PX - 1);
    }
#endif

    // Checks whether the given tile is masked by a filled tile in front of it
    __device__ bool isMasked(uint32_t t) const {
        uint3 i = unpack(t);
        return filled(t) >= (i.z * TILE_SIZE_PX + TILE_SIZE_PX - 1);
    }

    const uint32_t per_side;
    const uint32_t total;
    uint32_t* __restrict__ const data;
};
