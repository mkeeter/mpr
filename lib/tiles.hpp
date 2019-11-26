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
          data(CUDA_MALLOC(uint32_t, 2 * total + pow(per_side, 2))),
          terminal(CUDA_MALLOC(uint8_t, total))
    {
        reset();
    }

    ~Tiles() {
        CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaFree(terminal));
    }

    __device__ uint3 lowerCornerVoxel(uint32_t t) const {
        const auto i = unpack(t);
        return make_uint3(i.x * TILE_SIZE_PX,
                          i.y * TILE_SIZE_PX,
                          i.z * TILE_SIZE_PX);
    }

    __device__ uint32_t headAtVoxel(uint3 p) const {
        const uint32_t t =
            (p.x / TILE_SIZE_PX) +
            (p.y / TILE_SIZE_PX) * per_side +
            (p.y / TILE_SIZE_PX) * per_side * per_side;
        return head(t);
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
        return make_float3(2.0f * ((f.x / per_side) - 0.5f),
                           2.0f * ((f.y / per_side) - 0.5f),
                           2.0f * ((f.z / per_side) - 0.5f));
    }

    __device__ float3 tileToUpperPos(uint32_t t) const {
        const auto f = unpackFloat(t);
        return make_float3(2.0f * ((f.x + 1.0f) / per_side - 0.5f),
                           2.0f * ((f.y + 1.0f) / per_side - 0.5f),
                           2.0f * ((f.z + 1.0f) / per_side - 0.5f));
    }

    // Returns the z index of the filled tile at t, if present
    //
    // This ignores Z coordinates, because filled tiles occlude
    // anything behind them, so only the highest Z value matters.
    __device__ uint32_t& filled(uint32_t t) {
        uint3 i = unpack(t);
        return data[total*2 + i.x + i.y * per_side];
    }
    __device__ uint32_t filled(uint32_t t) const {
        uint3 i = unpack(t);
        return data[total*2 + i.x + i.y * per_side];
    }

    // Returns the Z height of the given pixels
    __host__ __device__ uint32_t filledAt(uint32_t px, uint32_t py) const {
        const auto tx = px / TILE_SIZE_PX;
        const auto ty = py / TILE_SIZE_PX;
        return data[total*2 + tx + ty * per_side];
    }

    // Returns the tile index of the i'th active tile
    __host__ __device__ uint32_t active(uint32_t i) const
        { return data[i]; }
    __host__ __device__ uint32_t& active(uint32_t i)
        { return data[i]; }

    // Returns the subtape head of the tile t
    __host__ __device__ uint32_t head(uint32_t t) const
        { return data[total + t]; }
    __host__ __device__ uint32_t& head(uint32_t t)
        { return data[total + t]; }

#ifdef __CUDACC__
    // Marks that the tile at t is filled.  Filled tiles occlude
    // everything behind them, so this is an atomicMax operation.
    __device__ void insertFilled(uint32_t t) {
        uint3 i = unpack(t);
        if (DIMENSION == 2) {
            atomicMax(&filled(t), i.z + 1);
        } else {
            atomicMax(&filled(t), i.z * TILE_SIZE_PX + TILE_SIZE_PX - 1);
        }
    }
    __device__ void insertActive(uint32_t t) {
        active(atomicAdd(&num_active, 1)) = t;
    }

    // Checks whether the given tile is masked by a filled tile in front of it
    __device__ bool isMasked(uint32_t t) const {
        uint3 i = unpack(t);
        return filled(t) >= (i.z * TILE_SIZE_PX + TILE_SIZE_PX - 1);
    }
#endif

    void reset() {
        num_active = 0;
        num_filled = 0;
        cudaMemset(&data[total], 0,
                   sizeof(uint32_t) * (total + pow(per_side, 2)));
    }

    const uint32_t per_side;
    const uint32_t total;

    uint32_t num_active;
    uint32_t num_filled;

    uint8_t* __restrict__ const terminal;
protected:
    uint32_t* __restrict__ const data;
};
