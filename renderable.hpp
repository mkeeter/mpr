#pragma once
#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>
#include "tape.hpp"
#include "parameters.hpp"
#include "check.hpp"

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

    struct Subtapes {
        /*  Each subtape is an array of LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE clauses.
         *
         *  subtape.data[i]'s valid clauses go from start[i] to the end of the
         *  array; if it's not full, the beginning is invalid (which is weird,
         *  but makes sense if you know how we're constructing subtapes). */
        uint32_t start[LIBFIVE_CUDA_NUM_SUBTAPES];
        uint32_t next[LIBFIVE_CUDA_NUM_SUBTAPES];
        Clause   data[LIBFIVE_CUDA_NUM_SUBTAPES]
                     [LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE];
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

    // This is a block of data which should be indexed as i[threadIdx.x]
    using IntervalRegisters = Interval[LIBFIVE_CUDA_TILE_THREADS];
    using FloatRegisters = float[LIBFIVE_CUDA_TILE_SIZE_PX *
                                 LIBFIVE_CUDA_TILE_SIZE_PX];
    using ChoiceArray = uint8_t[LIBFIVE_CUDA_TILE_THREADS];

    // regs_i and regs_f are both stored in scratch, to reduce
    // total memory usage (since we're only using one or the other)
    uint8_t* const scratch;
    IntervalRegisters* __restrict__ const regs_i;
    FloatRegisters* __restrict__ const regs_f;

    ChoiceArray* __restrict__ const csg_choices;

    struct Tiles {
        Tiles(uint32_t count)
            : data(CUDA_MALLOC(uint32_t, 2 * count)), size(2 * count)
        {
            reset();
        }

        ~Tiles() {
            CHECK(cudaFree(data));
        }

        __host__ __device__ uint32_t filled(uint32_t i) const
            { return data[size - i - 1]; }
        __host__ __device__ uint32_t active(uint32_t i) const
            { return data[i * 2]; }
        __host__ __device__ uint32_t head(uint32_t i) const
            { return data[i * 2 + 1]; }

        __host__ __device__ uint32_t& filled(uint32_t i)
            { return data[size - i - 1]; }
        __host__ __device__ uint32_t& active(uint32_t i)
            { return data[i * 2]; }
        __host__ __device__ uint32_t& head(uint32_t i)
            { return data[i * 2 + 1]; }

        __device__ void insert_filled(uint32_t index);
        __device__ void insert_active(uint32_t index);

        void reset() {
            num_active = 0;
            num_filled = 0;
            num_subtapes = 1;
        }

        uint32_t num_active;
        uint32_t num_filled;

        Subtapes subtapes;
        uint32_t num_subtapes;
    protected:
        uint32_t* __restrict__ const data;
        const uint32_t size;
    };

    Tiles tiles;

    uint8_t* __restrict__ const image;

    __device__ void processTiles(const uint32_t offset, const View& v);
    __device__ void drawFilledTile(const uint32_t tile);
    __device__ void drawAmbiguousTile(const uint32_t tile,
                                      const uint32_t subtape_index,
                                      const View& v);
    __device__ void buildSubtapes(const uint32_t offset);

    cudaStream_t streams[2];
protected:
    static size_t floatRegSize(uint16_t num_regs);
    static size_t intervalRegSize(uint16_t num_regs);

    Renderable(libfive::Tree tree, uint32_t image_size_px);

    Renderable(const Renderable& other)=delete;
    Renderable& operator=(const Renderable& other)=delete;
};
