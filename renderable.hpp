#pragma once
#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>
#include "tape.hpp"
#include "subtape.hpp"

struct Interval;

class Renderable {
public:
    class Deleter {
    public:
        void operator()(Renderable* r);
    };

    struct View {
        float center[2];
        float scale;
    };

    using Handle = std::unique_ptr<Renderable, Deleter>;

    // Returns a GPU-allocated Renderable struct
    static Handle build(libfive::Tree tree, uint32_t image_size_px);
    ~Renderable();
    void run(const View& v);

    Tape tape;

    // Render parameters
    const uint32_t IMAGE_SIZE_PX;
    const uint32_t TILE_COUNT;
    const uint32_t TOTAL_TILES;

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

    __device__ void processTiles(const uint32_t offset, const View& v);
    __device__ void drawFilledTiles(const uint32_t offset, const View& v);
    __device__ void drawAmbiguousTiles(const uint32_t offset, const View& v);
    __device__ void buildSubtapes(const uint32_t offset);

    cudaStream_t streams[2];
protected:
    Renderable(libfive::Tree tree, uint32_t image_size_px);

    Renderable(const Renderable& other)=delete;
    Renderable& operator=(const Renderable& other)=delete;
};
