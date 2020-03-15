#include "util.hpp"

struct Tile {
    uint32_t head;
    uint32_t pos;   // absolute position
    uint32_t index; // index of this tile in the active list
};

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
struct Tiles {
    Tiles(const uint32_t image_size_px)
        : per_side(image_size_px / TILE_SIZE_PX),
          data(nullptr), size(0)
    {
        reset();
    }

    __host__ __device__
    constexpr static uint32_t sizePx() {
        return TILE_SIZE_PX;
    }

    constexpr static uint32_t dimension() {
        return DIMENSION;
    }

    ~Tiles() {
        CUDA_FREE(data);
    }

    void resizeToFit(uint32_t num) {
        if (num > size) {
            CUDA_FREE(data);
            data = CUDA_MALLOC(Tile, num);
            size = num;
        }
        for (unsigned i=0; i < size; ++i) {
            data[i].head = 0;
        }
    }

    void setDefaultPositions() {
        for (unsigned i=0; i < size; ++i) {
            data[i].pos = i;
        }
    }

    __device__ inline uint3 tilePos(uint32_t t) const {
        const auto i = data[t].pos;
        return make_uint3(
            i % per_side,
            (i / per_side) % per_side,
            (i / per_side) / per_side);
    }

    __device__ inline float3 tilePosFloat(uint32_t t) const {
        const auto i = tilePos(t);
        return make_float3(i.x, i.y, i.z);
    }

    __device__ inline uint3 lowerCornerVoxel(uint32_t t) const {
        const auto i = tilePos(t);
        return make_uint3(i.x * TILE_SIZE_PX,
                          i.y * TILE_SIZE_PX,
                          i.z * TILE_SIZE_PX);
    }

    __device__ inline float3 tileToLowerPos(uint32_t t) const {
        const auto f = tilePosFloat(t);
        return make_float3(2.0f * ((f.x / per_side) - 0.5f),
                           2.0f * ((f.y / per_side) - 0.5f),
                           2.0f * ((f.z / per_side) - 0.5f));
    }

    __device__ inline float3 tileToUpperPos(uint32_t t) const {
        const auto f = tilePosFloat(t);
        return make_float3(2.0f * ((f.x + 1.0f) / per_side - 0.5f),
                           2.0f * ((f.y + 1.0f) / per_side - 0.5f),
                           2.0f * ((f.z + 1.0f) / per_side - 0.5f));
    }

    // Returns the subtape head of the tile t
    __host__ __device__ inline uint32_t head(uint32_t t) const {
        return data[t].head & ((1U << 31) - 1);
    }
    __host__ __device__ inline void setHead(uint32_t t, uint32_t head, bool terminal) {
        data[t].head = head | (terminal << 31);
    }
    __host__ __device__ inline bool terminal(uint32_t t) const {
        return data[t].head & (1U << 31);
    }

    __host__ __device__ inline uint32_t& pos(uint32_t t) {
        return data[t].pos;
    }
    __host__ __device__ inline uint32_t pos(uint32_t t) const {
        return data[t].pos;
    }

    __host__ __device__ inline uint32_t& index(uint32_t t) {
        return data[t].index;
    }
    __host__ __device__ inline uint32_t index(uint32_t t) const {
        return data[t].index;
    }

    void reset() {
        cudaMemset(data, 0, sizeof(Tile) * size);
    }

    const uint32_t per_side;
    uint32_t size;  // Amount of allocated tiles
protected:
    Tile* __restrict__ data;
};
