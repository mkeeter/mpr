#include "renderable.hpp"

__constant__ static uint64_t const_buffer[0x2000];

////////////////////////////////////////////////////////////////////////////////

template <typename IntervalRegisters, typename ChoiceArray>
__device__
Interval walkI(
        const Clause* __restrict__ clause_ptr,
        const float* __restrict__ constant_ptr,
        const uint32_t num_clauses,
        IntervalRegisters* const __restrict__ regs,
        ChoiceArray* const __restrict__ choices)
{
    using namespace libfive::Opcode;

    uint32_t choice_index = 0;

    // We copy a chunk of the tape from constant to shared memory, with
    // each thread moving two Clause in a SIMD operation.
    constexpr unsigned SHARED_CLAUSE_SIZE = LIBFIVE_CUDA_TILE_THREADS * 2;
    __shared__ Clause clauses[SHARED_CLAUSE_SIZE];

    for (uint32_t i=0; i < num_clauses; ++i) {
        if ((i % SHARED_CLAUSE_SIZE) == 0) {
            const uint32_t j = threadIdx.x * 2;
            __syncthreads();
            *reinterpret_cast<uint4*>(&clauses[j]) =
                *reinterpret_cast<const uint4*>(&clause_ptr[i + j]);
            __syncthreads();
        }

        const Clause c = clauses[i % SHARED_CLAUSE_SIZE];
        // All clauses must have at least one argument, since constants
        // and VAR_X/Y/Z are handled separately.
        Interval lhs;
        if (c.banks & 1) {
            const float f = constant_ptr[c.lhs];
            lhs.lower = f;
            lhs.upper = f;
        } else {
            lhs = regs[c.lhs][threadIdx.x];
        }

        Interval rhs;
        if (c.opcode >= OP_ADD) {
            if (c.banks & 2) {
                const float f = constant_ptr[c.rhs];
                rhs.lower = f;
                rhs.upper = f;
            } else {
                rhs = regs[c.rhs][threadIdx.x];
            }
        }

        Interval out;
        switch (c.opcode) {
            case OP_SQUARE: out = lhs.square(); break;
            case OP_SQRT: out = lhs.sqrt(); break;
            case OP_NEG: out = -lhs; break;
            // Skipping transcendental functions for now

            case OP_ADD: out = lhs + rhs; break;
            case OP_MUL: out = lhs * rhs; break;
            case OP_DIV: out = lhs / rhs; break;
            case OP_MIN: if (lhs.upper < rhs.lower) {
                             choices[choice_index][threadIdx.x] = 1;
                             out = lhs;
                         } else if (rhs.upper < lhs.lower) {
                             choices[choice_index][threadIdx.x] = 2;
                             out = rhs;
                         } else {
                             choices[choice_index][threadIdx.x] = 0;
                             out = lhs.min(rhs);
                         }
                         choice_index++;
                         break;
            case OP_MAX: if (lhs.lower > rhs.upper) {
                             choices[choice_index][threadIdx.x] = 1;
                             out = lhs;
                         } else if (rhs.lower > lhs.upper) {
                             choices[choice_index][threadIdx.x] = 2;
                             out = rhs;
                         } else {
                             choices[choice_index][threadIdx.x] = 0;
                             out = lhs.max(rhs);
                         }
                         choice_index++;
                         break;
            case OP_SUB: out = lhs - rhs; break;

            // Skipping various hard functions here
            default: break;
        }

        regs[c.out][threadIdx.x] = out;
    }
    // Copy output to standard register before exiting
    const Clause c = clause_ptr[num_clauses - 1];
    return regs[c.out][threadIdx.x];
#undef STORE_LOCAL_CLAUSES
}

////////////////////////////////////////////////////////////////////////////////

template <typename FloatRegisters>
__device__
float walkF(const uint32_t index, const Tape& tape,
            const Subtapes& subtapes, uint32_t subtape_index,
            FloatRegisters* const __restrict__ regs)
{
    assert(subtape_index != 0);
    using namespace libfive::Opcode;

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);

    uint32_t s = subtapes.start[subtape_index];

    __shared__ Clause clauses[LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE];
#define STORE_LOCAL_CLAUSES() do {                                      \
        for (unsigned i=index; i < LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;     \
                               i += blockDim.x * blockDim.y)            \
        {                                                               \
            clauses[i] = subtapes.data[subtape_index][i];               \
        }                                                               \
        __syncthreads();                                                \
    } while (0)

    STORE_LOCAL_CLAUSES();

    while (true) {
        if (s == LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
            const uint32_t next = subtapes.next[subtape_index];
            if (next) {
                subtape_index = next;
                s = subtapes.start[subtape_index];
            } else {
                return regs[clauses[s - 1].out][index];
            }
            __syncthreads();
            STORE_LOCAL_CLAUSES();
        }

        const Clause c = clauses[s++];

        // All clauses must have at least one argument, since constants
        // and VAR_X/Y/Z are handled separately.
        float lhs;
        if (c.banks & 1) {
            lhs = constant_ptr[c.lhs];
        } else {
            lhs = regs[c.lhs][index];
        }

        float rhs;
        if (c.opcode >= OP_ADD) {
            if (c.banks & 2) {
                rhs = constant_ptr[c.rhs];
            } else {
                rhs = regs[c.rhs][index];
            }
        }

        float out;
        switch (c.opcode) {
            case OP_SQUARE: out = lhs * lhs; break;
            case OP_SQRT: out = sqrtf(lhs); break;
            case OP_NEG: out = -lhs; break;
            // Skipping transcendental functions for now

            case OP_ADD: out = lhs + rhs; break;
            case OP_MUL: out = lhs * rhs; break;
            case OP_DIV: out = lhs / rhs; break;
            case OP_MIN: out = fminf(lhs, rhs); break;
            case OP_MAX: out = fmaxf(lhs, rhs); break;
            case OP_SUB: out = lhs - rhs; break;

            // Skipping various hard functions here
            default: break;
        }
        regs[c.out][index] = out;
    }
    assert(false);
    return 0.0f;

#undef STORE_LOCAL_CLAUSES
}

////////////////////////////////////////////////////////////////////////////////

template <typename ChoiceArray, typename ActiveArray>
__device__
void pushSubtape(const uint32_t index, const uint32_t tile_index,
                 uint32_t num_clauses,
                 const Clause* __restrict__ const clause_ptr,
                 ActiveArray* __restrict__ const active,
                 ChoiceArray* __restrict__& choices,
                 uint32_t& subtape_index, uint32_t& s,
                 Tiles& out)
{
    // Walk from the root of the tape downwards
    while (num_clauses--) {
        using namespace libfive::Opcode;
        Clause c = clause_ptr[num_clauses];
        if (active[c.out][index]) {
            active[c.out][index] = false;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
                const uint8_t choice = (*(--choices))[tile_index];
                if (choice == 1) {
                    if (!(c.banks & 1)) {
                        active[c.lhs][index] = true;
                        if (c.lhs == c.out) {
                            continue;
                        }
                        c.rhs = c.lhs;
                    }
                } else if (choice == 2) {
                    if (!(c.banks & 2)) {
                        active[c.rhs][index] = true;
                        if (c.rhs == c.out) {
                            continue;
                        }
                        c.lhs = c.rhs;
                    }
                } else if (choice == 0) {
                    if (!(c.banks & 1)) {
                        active[c.lhs][index] = true;
                    }
                    if (!(c.banks & 2)) {
                        active[c.rhs][index] = true;
                    }
                } else {
                    assert(false);
                }
            } else {
                if (c.opcode >= OP_SQUARE && !(c.banks & 1)) {
                    active[c.lhs][index] = true;
                }
                if (c.opcode >= OP_ADD && !(c.banks & 2)) {
                    active[c.rhs][index] = true;
                }
            }

            // Allocate a new subtape and begin writing to it
            if (s == 0) {
                auto next_subtape_index = atomicAdd(&out.num_subtapes, 1);
                out.subtapes.start[subtape_index] = 0;
                out.subtapes.next[next_subtape_index] = subtape_index;

                subtape_index = next_subtape_index;
                s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
            }
            out.subtapes.data[subtape_index][--s] = c;
        } else if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            --choices;
        }
    }
}
////////////////////////////////////////////////////////////////////////////////

TileRenderer::TileRenderer(const Tape& tape, Image& image)
    : tape(tape), image(image),
      tiles(image.size_px, LIBFIVE_CUDA_TILE_SIZE_PX),

      regs(CUDA_MALLOC(IntervalRegisters, LIBFIVE_CUDA_TILE_BLOCKS *
                                          sizeof(Interval) * tape.num_regs)),
      active(CUDA_MALLOC(ActiveArray, LIBFIVE_CUDA_TILE_BLOCKS *
                                      tape.num_regs)),
      choices(tape.num_csg_choices ?
              CUDA_MALLOC(ChoiceArray,
                LIBFIVE_CUDA_TILE_BLOCKS * tape.num_csg_choices *
                num_passes())
              : nullptr)
{
    // Nothing to do here
}

TileRenderer::~TileRenderer()
{
    CHECK(cudaFree(regs));
    CHECK(cudaFree(choices));
}

size_t TileRenderer::num_passes() const {
    const auto denom = LIBFIVE_CUDA_TILE_THREADS * LIBFIVE_CUDA_TILE_BLOCKS;
    return (tiles.total + denom - 1) / denom;
}

__device__
void TileRenderer::check(const uint32_t tile, const View& v)
{
    auto regs = this->regs + tape.num_regs * blockIdx.x;
    {   // Prepopulate axis values
        const float x = tile / tiles.per_side;
        const float y = tile % tiles.per_side;

        Interval vs[3];
        const Interval X = {x / tiles.per_side, (x + 1) / tiles.per_side};
        vs[0].lower = 2.0f * (X.lower - 0.5f - v.center[0]) * v.scale;
        vs[0].upper = 2.0f * (X.upper - 0.5f - v.center[0]) * v.scale;

        const Interval Y = {y / tiles.per_side, (y + 1) / tiles.per_side};
        vs[1].lower = 2.0f * (Y.lower - 0.5f - v.center[1]) * v.scale;
        vs[1].upper = 2.0f * (Y.upper - 0.5f - v.center[1]) * v.scale;

        vs[2].lower = 0.0f;
        vs[2].upper = 0.0f;

        for (unsigned i=0; i < 3; ++i) {
            if (tape.axes.reg[i] != UINT16_MAX) {
                regs[tape.axes.reg[i]][threadIdx.x] = vs[i];
            }
        }
    }

    // Unpack a 1D offset into the data arrays
    auto csg_choices = choices + tile / LIBFIVE_CUDA_TILE_THREADS
                                      * tape.num_csg_choices;

    // Run actual evaluation
    const Interval result = walkI(&tape[0], &tape.constant(0),
                                  tape.num_clauses, regs, csg_choices);

    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper < 0.0f) {
        tiles.insert_filled(tile);
    }

    // If the tile is ambiguous, then record it as needing further refinement
    else if (result.lower <= 0.0f && result.upper >= 0.0f) {
        tiles.insert_active(tile);
    }
}

__global__ void TileRenderer_check(TileRenderer* r,
                                   const uint32_t offset,
                                   View v)
{
    // This should be a 1D kernel
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t tile = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (tile < r->tiles.total) {
        r->check(tile, v);
    }
}

__device__ uint32_t TileRenderer::buildTape(const uint32_t tile)
{
    // Pick a subset of the active array to use for this block
    auto active = this->active + blockIdx.x * tape.num_regs;
    const uint32_t index = threadIdx.x;

    for (uint32_t r=0; r < tape.num_regs; ++r) {
        active[r][index] = false;
    }

    // Pick an offset CSG choices array, pointing to the last
    // set of choices (we'll walk this back as we process the tape)
    auto choices = (this->choices + tile / LIBFIVE_CUDA_TILE_THREADS
                                             * tape.num_csg_choices)
                    + tape.num_csg_choices;

    // Mark the root of the tree as true
    uint32_t num_clauses = tape.num_clauses;
    active[tape[num_clauses - 1].out][index] = true;

    // Claim a subtape to populate
    uint32_t subtape_index = atomicAdd(&tiles.num_subtapes, 1);
    assert(subtape_index < LIBFIVE_CUDA_NUM_SUBTAPES);

    // Since we're reversing the tape, this is going to be the
    // end of the linked list (i.e. next = 0)
    tiles.subtapes.next[subtape_index] = 0;
    uint32_t s = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;

    // Walk from the root of the tape downwards
    pushSubtape(index, tile % LIBFIVE_CUDA_TILE_THREADS, num_clauses,
                &tape[0], active, choices, subtape_index, s, tiles);

    // The last subtape may not be completely filled
    tiles.subtapes.start[subtape_index] = s;

    return subtape_index;
}

__global__ void TileRenderer_buildTape(TileRenderer* r, const uint32_t offset)
{
    // This is a 1D kernel which consumes a tile and writes out its tape
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (i < r->tiles.num_active) {
        // Pick out the next active tile
        const uint32_t tile = r->tiles.active(i);

        // Store the linked list of subtapes into the active tiles list
        r->tiles.head(i) = r->buildTape(tile);
    }
}

__device__ void TileRenderer::drawFilled(const uint32_t tile)
{
    static_assert(LIBFIVE_CUDA_TILE_SIZE_PX >= 16, "Tiles are too small");
    static_assert(LIBFIVE_CUDA_TILE_SIZE_PX % 16 == 0, "Invalid tile size");

    // Convert from tile position to pixels
    const uint32_t px = (tile / tiles.per_side) * LIBFIVE_CUDA_TILE_SIZE_PX;
    const uint32_t py = (tile % tiles.per_side) * LIBFIVE_CUDA_TILE_SIZE_PX;

    uint4* pix = reinterpret_cast<uint4*>(&image[px + py * image.size_px]);
    const uint4 fill = make_uint4(0xD0D0D0D0, 0xD0D0D0D0, 0xD0D0D0D0, 0xD0D0D0D0);
    for (unsigned y=0; y < LIBFIVE_CUDA_TILE_SIZE_PX; y++) {
        for (unsigned x=0; x < LIBFIVE_CUDA_TILE_SIZE_PX; x += 16) {
            *pix = fill;
            pix++;
        }
        pix += (image.size_px - LIBFIVE_CUDA_TILE_SIZE_PX) / 16;
    }
}

__global__ void TileRenderer_drawFilled(TileRenderer* r, const uint32_t offset)
{
    // Each thread picks a block and fills in the whole thing
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t i = start + offset;
    if (i < r->tiles.num_filled) {
        const uint32_t tile = r->tiles.filled(i);
        r->drawFilled(tile);
    }
}

////////////////////////////////////////////////////////////////////////////////

PixelRenderer::PixelRenderer(const Tape& tape, Image& image)
    : tape(tape), image(image),
      regs(CUDA_MALLOC(FloatRegisters,
                       tape.num_regs * LIBFIVE_CUDA_RENDER_BLOCKS))
{
    // Nothing to do here
}

__device__ void PixelRenderer::draw(
        const uint32_t tile, const uint32_t tiles_per_side,
        const Subtapes& subtapes, const uint32_t subtape_index,
        const View& v)
{
    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;
    const uint32_t index = dx + dy * LIBFIVE_CUDA_SUBTILE_SIZE_PX;

    // Pick an index into the register array
    auto regs = this->regs + tape.num_regs * blockIdx.x;

    // Convert from tile position to pixels
    uint32_t px = (tile / tiles_per_side) * LIBFIVE_CUDA_SUBTILE_SIZE_PX + dx;
    uint32_t py = (tile % tiles_per_side) * LIBFIVE_CUDA_SUBTILE_SIZE_PX + dy;

    {   // Prepopulate axis values
        const float x = px / (image.size_px - 1.0f);
        const float y = py / (image.size_px - 1.0f);
        float vs[3];
        vs[0] = 2.0f * (x - 0.5f - v.center[0]) * v.scale;
        vs[1] = 2.0f * (y - 0.5f - v.center[1]) * v.scale;
        vs[2] = 0.0f;
        for (unsigned i=0; i < 3; ++i) {
            if (tape.axes.reg[i] != UINT16_MAX) {
                regs[tape.axes.reg[i]][index] = vs[i];
            }
        }
    }
    const float f = walkF(index, tape, subtapes, subtape_index, regs);
    if (f < 0.0f) {
        image(px, py) = 255;
    }
}

__global__ void PixelRenderer_draw(PixelRenderer* r,
                                   const Tiles& tiles,
                                   const uint32_t offset, View v)
{
    // We assume one thread per pixel in a tile
    assert(blockDim.x == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(blockDim.y == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    // Pick an active tile from the list
    const uint32_t i = offset + blockIdx.x;
    if (i < tiles.num_active) {
        const uint32_t tile = tiles.active(i);
        const uint32_t subtape_index = tiles.head(i);

        r->draw(tile, tiles.per_side, tiles.subtapes, subtape_index, v);
    }
}

////////////////////////////////////////////////////////////////////////////////

void Renderable::Deleter::operator()(Renderable* r)
{
    r->~Renderable();
    CHECK(cudaFree(r));
}

Renderable::~Renderable()
{
    for (auto& s : streams) {
        CHECK(cudaStreamDestroy(s));
    }
}

Renderable::Handle Renderable::build(libfive::Tree tree, uint32_t image_size_px)
{
    auto out = CUDA_MALLOC(Renderable, 1);
    new (out) Renderable(tree, image_size_px);
    return Handle(out);
}

Renderable::Renderable(libfive::Tree tree, uint32_t image_size_px)
    : image(image_size_px),
      tape(std::move(Tape::build(tree))),

      tile_renderer(tape, image),
      pixel_renderer(tape, image)
{
    CHECK(cudaStreamCreate(&streams[0]));
    CHECK(cudaStreamCreate(&streams[1]));
}

void Renderable::run(const View& view)
{
    cudaStream_t streams[2] = {this->streams[0], this->streams[1]};

    // Record this local variable because otherwise it looks up memory
    // that has been loaned to the GPU and not synchronized.
    TileRenderer* tile_renderer = &this->tile_renderer;
    const uint32_t total_tiles = tile_renderer->tiles.total;
    const uint32_t stride = LIBFIVE_CUDA_TILE_THREADS *
                            LIBFIVE_CUDA_TILE_BLOCKS;
    PixelRenderer* pixel_renderer = &this->pixel_renderer;

    // Reset everything in preparation for a render
    tile_renderer->tiles.reset();
    cudaMemset(image.data, 0, image.size_px * image.size_px);

    tape.sendToConstantMemory((const char*)const_buffer);

    // Do per-tile evaluation to get filled / ambiguous tiles
    for (unsigned i=0; i < total_tiles; i += stride) {
        TileRenderer_check<<<LIBFIVE_CUDA_TILE_BLOCKS,
                             LIBFIVE_CUDA_TILE_THREADS,
                             0, streams[0]>>>(tile_renderer, i, view);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaStreamSynchronize(streams[0]));

    // Pull a few variables back from the GPU
    const uint32_t filled_tiles = tile_renderer->tiles.num_filled;
    const uint32_t active_tiles = tile_renderer->tiles.num_active;

    for (unsigned i=0; i < filled_tiles; i += stride) {
        // Drawing filled and ambiguous tiles can happen simultaneously,
        // so we assign each one to a separate stream.
        TileRenderer_drawFilled<<<LIBFIVE_CUDA_TILE_BLOCKS,
                                  LIBFIVE_CUDA_TILE_THREADS,
                                  0, streams[1]>>>(tile_renderer, i);
        CHECK(cudaGetLastError());
    }

    // Build subtapes in memory for ambiguous tiles
    for (unsigned i=0; i < active_tiles; i += stride) {
        TileRenderer_buildTape<<<LIBFIVE_CUDA_TILE_BLOCKS,
                                 LIBFIVE_CUDA_TILE_THREADS,
                                 0, streams[0]>>>(tile_renderer, i);
        CHECK(cudaGetLastError());
    }

    // Do pixel-by-pixel rendering for ambiguous tiles
    for (unsigned i=0; i < active_tiles; i += LIBFIVE_CUDA_RENDER_BLOCKS) {
        const dim3 T(LIBFIVE_CUDA_SUBTILE_SIZE_PX, LIBFIVE_CUDA_SUBTILE_SIZE_PX);
        PixelRenderer_draw<<<LIBFIVE_CUDA_RENDER_BLOCKS,
                             T, 0, streams[0]>>>(
            pixel_renderer, tile_renderer->tiles, i, view);
        CHECK(cudaGetLastError());
    }
}
