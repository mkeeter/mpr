#pragma once
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>

#include "check.hpp"
#include "clause.hpp"
#include "filled.hpp"
#include "gpu_interval.hpp"
#include "gpu_deriv.hpp"
#include "image.hpp"
#include "parameters.hpp"
#include "queue.hpp"
#include "subtapes.hpp"
#include "tape.hpp"
#include "tiles.hpp"
#include "view.hpp"

enum TileResult {
    TILE_FILLED,
    TILE_EMPTY,
    TILE_AMBIGUOUS,
};

template <unsigned TILE_SIZE_PX, unsigned DIMENSION>
class TileRenderer {
public:
    TileRenderer(const Tape& tape, Subtapes& subtapes, Image& image);

    // Evaluates the given tile.
    //      Filled -> Pushes it to the list of filed tiles
    //      Ambiguous -> Pushes it to the list of active tiles and builds tape
    //      Empty -> Does nothing
    //  Reverses the tapes
    __device__ TileResult check(const uint32_t tile, const View& v);

    const Tape& tape;

    Tiles<TILE_SIZE_PX, DIMENSION> tiles;

protected:
    Subtapes& subtapes;

    TileRenderer(const TileRenderer& other)=delete;
    TileRenderer& operator=(const TileRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

template <unsigned TILE_SIZE_PX, unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
class SubtileRenderer {
public:
    SubtileRenderer(const Tape& tape, Subtapes& subtapes, Image& image,
                    Tiles<TILE_SIZE_PX, DIMENSION>& prev);

    constexpr static unsigned __host__ __device__ subtilesPerTileSide() {
        static_assert(TILE_SIZE_PX % SUBTILE_SIZE_PX == 0,
                      "Cannot evenly divide tiles into subtiles");
        return TILE_SIZE_PX / SUBTILE_SIZE_PX;
    }
    constexpr static unsigned __host__ __device__ subtilesPerTile() {
        return pow(subtilesPerTileSide(), DIMENSION);
    }

    // Same functions as in TileRenderer, but these take a subtape because
    // they're refining a tile into subtiles
    __device__ TileResult check(
            const uint32_t subtile,
            const uint32_t tile,
            const View& v);

    const Tape& tape;

    // Reference to tiles generated in previous stage
    Tiles<TILE_SIZE_PX, DIMENSION>& tiles;

    // New tiles generated in this stage
    Tiles<SUBTILE_SIZE_PX, DIMENSION> subtiles;

protected:
    Subtapes& subtapes;

    SubtileRenderer(const SubtileRenderer& other)=delete;
    SubtileRenderer& operator=(const SubtileRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

template <unsigned SUBTILE_SIZE_PX, unsigned DIMENSION>
class PixelRenderer {
public:
    PixelRenderer(const Tape& tape, const Subtapes& subtapes, Image& image,
                  const Tiles<SUBTILE_SIZE_PX, DIMENSION>& prev);

    constexpr static unsigned __host__ __device__ pixelsPerSubtile() {
        return pow(SUBTILE_SIZE_PX, DIMENSION);
    }

    // Draws the given tile, starting from the given subtape
    __device__ void draw(const uint32_t subtile, const View& v);

    const Tape& tape;
    Image& image;

    // Reference to tiles generated in previous stage
    const Tiles<SUBTILE_SIZE_PX, DIMENSION>& subtiles;

protected:
    const Subtapes& subtapes;

    PixelRenderer(const PixelRenderer& other)=delete;
    PixelRenderer& operator=(const PixelRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class Renderable; // forward declaration
class NormalRenderer {
public:
    NormalRenderer(const Tape& tape, const Subtapes& subtapes, Image& norm);

    // Evaluates the given location + subtape + view, returning the
    // packed normal result
    __device__ uint32_t draw(const float3 f, uint32_t subtape_index, const View& v);

    const Tape& tape;
    const Subtapes& subtapes;
    Image& norm;
protected:
    NormalRenderer(const NormalRenderer& other)=delete;
    NormalRenderer& operator=(const NormalRenderer& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class Renderable {
public:
    class Deleter {
    public:
        void operator()(Renderable* r);
    };
    using Handle = std::unique_ptr<Renderable, Deleter>;

    virtual ~Renderable();

    // Returns a GPU-allocated Renderable struct
    static Handle build(libfive::Tree tree, uint32_t image_size_px,
                        uint8_t dimension);
    virtual void run(const View& v)=0;

    uint32_t heightAt(const uint32_t x, const uint32_t y) const {
        return image(x, y);
    }
    virtual uint32_t normalAt(const uint32_t, const uint32_t) const {
        return 0;
    }

    static cudaGraphicsResource* registerTexture(GLuint t);
    virtual void copyToTexture(cudaGraphicsResource* gl_tex,
                               uint32_t texture_size,
                               bool append, bool mode)=0;
    virtual uint32_t dimension() const=0;

    Image image;
    Tape tape;

protected:
    cudaStream_t streams[LIBFIVE_CUDA_NUM_STREAMS];
    Subtapes subtapes;
    Queue queue_ping;
    Queue queue_pong;

    Renderable(libfive::Tree tree, uint32_t image_size_px);
    Renderable(const Renderable& other)=delete;
    Renderable& operator=(const Renderable& other)=delete;
};

////////////////////////////////////////////////////////////////////////////////

class Renderable3D : public Renderable {
public:
    void run(const View& v) override;
    void copyToTexture(cudaGraphicsResource* gl_tex,
                       uint32_t texture_size,
                       bool append, bool mode) override;

    __device__
    void copyDepthToSurface(cudaSurfaceObject_t surf,
                            uint32_t texture_size, bool append);

    __device__
    void copyDepthToImage();

    __device__
    void copyNormalToSurface(cudaSurfaceObject_t surf,
                             uint32_t texture_size, bool append);

    uint32_t normalAt(const uint32_t x, const uint32_t y) const override {
        return norm(x, y);
    }

    uint32_t dimension() const override { return 3; };

    // Returns the subtape head at the given voxel, or 0
    __device__
    uint32_t subtapeHeadAt(const uint3 v) const;

    __device__
    uint32_t drawNormals(const float3 f, const uint32_t subtape_index, const View& v);

    Image norm;

    Renderable3D(libfive::Tree tree, uint32_t image_size_px);

    Filled<64> filled_tiles;
    Filled<16> filled_subtiles;
    Filled<4>  filled_microtiles;

    TileRenderer<64, 3> tile_renderer;
    SubtileRenderer<64, 16, 3> subtile_renderer;
    SubtileRenderer<16, 4, 3> microtile_renderer;
    PixelRenderer<4, 3> pixel_renderer;
    NormalRenderer normal_renderer;

protected:
    Renderable3D(const Renderable3D& other)=delete;
    Renderable3D& operator=(const Renderable3D& other)=delete;

    friend class Renderable;
};

////////////////////////////////////////////////////////////////////////////////

class Renderable2D : public Renderable {
public:
    void run(const View& v) override;
    void copyToTexture(cudaGraphicsResource* gl_tex,
                       uint32_t texture_size,
                       bool append, bool mode) override;

    __device__
    void copyToSurface(cudaSurfaceObject_t surf,
                       uint32_t texture_size, bool append);

    __device__
    void copyDepthToImage();

    uint32_t dimension() const override { return 2; };

protected:
    Renderable2D(libfive::Tree tree, uint32_t image_size_px);

    Filled<64> filled_tiles;
    Filled<8>  filled_subtiles;

    TileRenderer<64, 2> tile_renderer;
    SubtileRenderer<64, 8, 2> subtile_renderer;
    PixelRenderer<8, 2> pixel_renderer;

    Renderable2D(const Renderable2D& other)=delete;
    Renderable2D& operator=(const Renderable2D& other)=delete;

    friend class Renderable;
};
