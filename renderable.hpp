#pragma once
#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>
#include "tape.hpp"
#include "subtape.hpp"

struct Interval;

class Renderable {
public:
    // Returns a GPU-allocated Renderable struct
    static Renderable* build(libfive::Tree tree,
            uint32_t image_size_px, uint32_t tile_size_px,
            uint32_t num_interval_blocks=8, uint32_t num_fill_blocks=1024,
            uint32_t num_subtapes=65536);
    void run(void);

    Tape tape;

    // Render parameters
    const uint32_t IMAGE_SIZE_PX;
    const uint32_t TILE_SIZE_PX;
    const uint32_t TILE_COUNT;
    const uint32_t TOTAL_TILES;
    const uint32_t NUM_INTERVAL_BLOCKS;
    const uint32_t THREADS_PER_INTERVAL_BLOCK;

    const uint32_t NUM_FILL_BLOCKS;
    const uint32_t NUM_SUBTAPES;

    // [regs_i, csg_choices] and regs_f are both stored in scratch, to reduce
    // total memory usage (since we're only using one or the other)
    uint8_t* const scratch;
    Interval* __restrict__ const regs_i;
    uint8_t* __restrict__ const csg_choices;
    float* __restrict__ const regs_f;

    uint32_t* __restrict__ const tiles;
    uint32_t active_tiles;
    uint32_t filled_tiles;

    Subtape* __restrict__ const subtapes;
    uint32_t active_subtapes;

    uint8_t* __restrict__ const image;

    __device__ void processTiles();
    __device__ void drawFilledTiles();
    __device__ void drawAmbiguousTiles();

    cudaStream_t streams[2];
protected:
    Renderable(libfive::Tree tree,
            uint32_t image_size_px, uint32_t tile_size_px,
            uint32_t num_interval_blocks, uint32_t num_fill_blocks,
            uint32_t num_subtapes);
};