#pragma once
#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>

#include "check.hpp"
#include "clause.hpp"
#include "gpu_interval.hpp"
#include "image.hpp"
#include "parameters.hpp"
#include "subtapes.hpp"
#include "tape.hpp"
#include "tiles.hpp"
#include "view.hpp"

class TileRenderer {
public:
    TileRenderer(const Tape& tape, Image& image);
    ~TileRenderer();

    // These are blocks of data which should be indexed as i[threadIdx.x]
    using IntervalRegisters = Interval[LIBFIVE_CUDA_TILE_THREADS];
    using ChoiceArray = uint8_t[LIBFIVE_CUDA_TILE_THREADS];
    using ActiveArray = uint8_t[LIBFIVE_CUDA_TILE_THREADS];

    // Evaluates the given tile.
    //      Filled -> Pushes it to the list of filed tiles
    //      Ambiguous -> Pushes it to the list of active tiles
    //      Empty -> Does nothing
    __device__ void check(const uint32_t tile, const View& v);

    // Builds a subtape for the given (active) tile, returning the head
    __device__ uint32_t buildTape(const uint32_t tile);

    // Fills in the given (filled) tile in the image
    __device__ void drawFilled(const uint32_t tile);

    const Tape& tape;
    Image& image;

    Tiles tiles;

protected:
    IntervalRegisters* __restrict__ const regs;
    ActiveArray* __restrict__ const active;
    ChoiceArray* __restrict__ const choices;

    size_t num_passes() const;

    TileRenderer(const TileRenderer& other)=delete;
    TileRenderer& operator=(const TileRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class SubtileRenderer {
public:
    // These are blocks of data which should be indexed as
    //      i[threadIdx.x + threadIdx.y * LIBFIVE_CUDA_SUBTILE_PER_TILE_SIDE]
    using IntervalRegisters = Interval[LIBFIVE_CUDA_SUBTILES_PER_TILE];
    using ChoiceArray = uint8_t[LIBFIVE_CUDA_SUBTILES_PER_TILE];

    // Same functions as in TileRenderer, but these take a subtape because
    // they're refining a tile into subtiles
    __device__ void check(
            const uint32_t tile, const Tape& tape,
            uint32_t subtape_index, const Subtapes& subtapes,
            const View& v);
    __device__ uint32_t buildTape(
            const uint32_t tile, const uint32_t subtape_index,
            const Subtapes& subtapes);
    __device__ void drawFilled(const uint32_t tile);

    Tiles tiles;

protected:
    IntervalRegisters* __restrict__ const regs;
    ChoiceArray* __restrict__ const choices;
    uint8_t* __restrict__ const image;

    SubtileRenderer(const SubtileRenderer& other)=delete;
    SubtileRenderer& operator=(const SubtileRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class PixelRenderer {
public:
    PixelRenderer(const Tape& tape, Image& image);

    using FloatRegisters = float[LIBFIVE_CUDA_PIXELS_PER_SUBTILE];

    // Draws the given tile, starting from the given subtape
    __device__ void draw(
            const uint32_t tile, const uint32_t total_tiles,
            const Subtapes& subtapes, const uint32_t subtape_index,
            const View& v);

protected:
    const Tape& tape;
    Image& image;

    FloatRegisters* __restrict__ const regs;

    PixelRenderer(const PixelRenderer& other)=delete;
    PixelRenderer& operator=(const PixelRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class Renderable {
public:
    class Deleter {
    public:
        void operator()(Renderable* r);
    };

    using Handle = std::unique_ptr<Renderable, Deleter>;

    // Returns a GPU-allocated Renderable struct
    static Handle build(libfive::Tree tree, uint32_t image_size_px);
    ~Renderable();
    void run(const View& v);

    Image image;

protected:
    Renderable(libfive::Tree tree, uint32_t image_size_px);

    Tape tape;

    cudaStream_t streams[2];
    TileRenderer tile_renderer;
    PixelRenderer pixel_renderer;

    Renderable(const Renderable& other)=delete;
    Renderable& operator=(const Renderable& other)=delete;
};
