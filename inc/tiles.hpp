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
          data(CUDA_MALLOC(uint32_t, total))
    {
        reset();
    }

    ~Tiles() {
        CUDA_CHECK(cudaFree(data));
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
            (p.z / TILE_SIZE_PX) * per_side * per_side;
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

    // Returns the subtape head of the tile t
    __host__ __device__ uint32_t head(uint32_t t) const {
        return data[t] & ((1U << 31) - 1);
    }
    __host__ __device__ void setHead(uint32_t t, uint32_t head, bool terminal) {
        data[t] = head | (terminal << 31);
    }
    __host__ __device__ bool terminal(uint32_t t) const {
        return data[t] & (1U << 31);
    }

    void reset() {
        cudaMemset(data, 0, sizeof(uint32_t) * total);
    }

    const uint32_t per_side;
    const uint32_t total;
protected:
    uint32_t* __restrict__ const data;
};
