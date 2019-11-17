#pragma once
#include <cuda_gl_interop.h>
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
    using Registers = float[LIBFIVE_CUDA_TILE_THREADS];
    using ChoiceArray = uint8_t[LIBFIVE_CUDA_TILE_THREADS];
    using ActiveArray = uint8_t[LIBFIVE_CUDA_TILE_THREADS];

    // Evaluates the given tile.
    //      Filled -> Pushes it to the list of filed tiles
    //      Ambiguous -> Pushes it to the list of active tiles and builds tape
    //      Empty -> Does nothing
    __device__ void check(const uint32_t tile, const View& v);

    // Fills in the given (filled) tile in the image
    __device__ void drawFilled(const uint32_t tile);

    // Reverses the given tile's tape
    __device__ void reverseTape(const uint32_t tile);

    const Tape& tape;
    Image& image;

    Tiles tiles;

protected:
    Registers* __restrict__ const regs_lower;
    Registers* __restrict__ const regs_upper;
    ActiveArray* __restrict__ const active;
    ChoiceArray* __restrict__ const choices;

    TileRenderer(const TileRenderer& other)=delete;
    TileRenderer& operator=(const TileRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class SubtileRenderer {
public:
    SubtileRenderer(const Tape& tape, Image& image, const TileRenderer& prev);
    ~SubtileRenderer();

    using Registers = float[LIBFIVE_CUDA_SUBTILES_PER_TILE *
                            LIBFIVE_CUDA_REFINE_TILES];
    using ActiveArray = uint8_t[LIBFIVE_CUDA_SUBTILE_THREADS];

    // Same functions as in TileRenderer, but these take a subtape because
    // they're refining a tile into subtiles
    __device__ void check(
            const uint32_t subtile,
            const uint32_t tile,
            const View& v);
    __device__ void drawFilled(const uint32_t tile);

    // Refines a tile tape into a subtile tape based on choices
    __device__ void buildTape(const uint32_t subtile,
                              const uint32_t tile);
    const Tape& tape;
    Image& image;

    const Tiles& tiles; // Reference to tiles generated in previous stage
    Tiles subtiles;     // New tiles generated in this stage

protected:
    uint8_t* const data;

    Registers* __restrict__ const regs_lower;
    Registers* __restrict__ const regs_upper;
    ActiveArray* __restrict__ const active;
    uint8_t* __restrict__ const choices;

    SubtileRenderer(const SubtileRenderer& other)=delete;
    SubtileRenderer& operator=(const SubtileRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class PixelRenderer {
public:
    PixelRenderer(const Tape& tape, Image& image, const SubtileRenderer& prev);
    ~PixelRenderer();

    using FloatRegisters = float[LIBFIVE_CUDA_PIXELS_PER_SUBTILE *
                                 LIBFIVE_CUDA_RENDER_SUBTILES];

    // Draws the given tile, starting from the given subtape
    __device__ void draw(const uint32_t subtile, const View& v);

protected:
    const Tape& tape;
    Image& image;
    const Tiles& subtiles; // Reference to tiles generated in previous stage

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

    static cudaGraphicsResource* registerTexture(GLuint t);
    void copyToTexture(cudaGraphicsResource* gl_tex, bool append);

    Image image;
    Tape tape;

protected:
    Renderable(libfive::Tree tree, uint32_t image_size_px);

    cudaStream_t streams[2];

    TileRenderer tile_renderer;
    SubtileRenderer subtile_renderer;
    PixelRenderer pixel_renderer;

    Renderable(const Renderable& other)=delete;
    Renderable& operator=(const Renderable& other)=delete;
};
