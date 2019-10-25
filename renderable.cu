#include "check.hpp"
#include "renderable.hpp"
#include "gpu_interval.hpp"
#include "parameters.hpp"

void Renderable::Deleter::operator()(Renderable* r)
{
    r->~Renderable();
    CHECK(cudaFree(r));
}

Renderable::~Renderable()
{
    CHECK(cudaFree(scratch));
    CHECK(cudaFree(tiles));
    CHECK(cudaFree(subtapes));
    CHECK(cudaFree(image));
    for (auto& s : streams) {
        CHECK(cudaStreamDestroy(s));
    }
}

Renderable::Handle Renderable::build(libfive::Tree tree, uint32_t image_size_px)
{
    auto out = CUDA_MALLOC(Renderable, 1);
    new (out) Renderable(tree, image_size_px);
    return std::unique_ptr<Renderable, Deleter>(out);
}

Renderable::Renderable(libfive::Tree tree, uint32_t image_size_px)
    : tape(std::move(Tape::build(tree))),

      IMAGE_SIZE_PX(image_size_px),
      TILE_COUNT(IMAGE_SIZE_PX / LIBFIVE_CUDA_TILE_SIZE_PX),
      TOTAL_TILES(TILE_COUNT * TILE_COUNT),

      scratch(CUDA_MALLOC(uint8_t,
          std::max(LIBFIVE_CUDA_TILE_BLOCKS * LIBFIVE_CUDA_TILE_THREADS *
                           sizeof(Interval) * tape.num_regs
                       + TOTAL_TILES * max(1, tape.num_csg_choices),
                   sizeof(float) * tape.num_regs * LIBFIVE_CUDA_RENDER_BLOCKS
                                 * LIBFIVE_CUDA_TILE_SIZE_PX
                                 * LIBFIVE_CUDA_TILE_SIZE_PX))),
      regs_i(reinterpret_cast<Interval*>(scratch)),
      csg_choices(scratch + LIBFIVE_CUDA_TILE_BLOCKS * LIBFIVE_CUDA_TILE_THREADS
                            * sizeof(Interval) * tape.num_regs),
      regs_f(reinterpret_cast<float*>(scratch)),

      tiles(CUDA_MALLOC(uint32_t, 2 * TOTAL_TILES)),
      active_tiles(0),
      filled_tiles(0),

      subtapes(CUDA_MALLOC(Subtape, LIBFIVE_CUDA_NUM_SUBTAPES)),
      active_subtapes(1),

      image(CUDA_MALLOC(uint8_t, IMAGE_SIZE_PX * IMAGE_SIZE_PX))
{
    cudaMemset(image, 0, IMAGE_SIZE_PX * IMAGE_SIZE_PX);
    CHECK(cudaStreamCreate(&streams[0]));
    CHECK(cudaStreamCreate(&streams[1]));
}

////////////////////////////////////////////////////////////////////////////////

__device__ void walkI(const Tape& tape,
                      const Interval X, const Interval Y,
                      Interval* const __restrict__ regs,
                      uint8_t* const __restrict__ choices)
{
    uint32_t choice_index = 0;
    for (uint32_t i=0; i < tape.num_clauses; ++i) {
        const Clause c = tape[i];
#define LHS ((!(c.banks & 1) ? regs[c.lhs] : Interval{tape.constant(c.lhs), \
                                                      tape.constant(c.lhs)}))
#define RHS ((!(c.banks & 2) ? regs[c.rhs] : Interval{tape.constant(c.rhs), \
                                                      tape.constant(c.rhs)}))
        using namespace libfive::Opcode;
        switch (c.opcode) {
            case VAR_X: regs[c.out] = X; break;
            case VAR_Y: regs[c.out] = Y; break;
            case VAR_Z: regs[c.out] = Interval{0.0f, 0.0f}; break;

            case OP_SQUARE: regs[c.out] = LHS.square(); break;
            case OP_SQRT: regs[c.out] = LHS.sqrt(); break;
            case OP_NEG: regs[c.out] = -LHS; break;
            // Skipping transcendental functions for now

            case OP_ADD: regs[c.out] = LHS + RHS; break;
            case OP_MUL: regs[c.out] = LHS * RHS; break;
            case OP_DIV: regs[c.out] = LHS / RHS; break;
            case OP_MIN: if (LHS.upper < RHS.lower) {
                             choices[choice_index] = 1;
                             regs[c.out] = LHS;
                         } else if (RHS.upper < LHS.lower) {
                             choices[choice_index] = 2;
                             regs[c.out] = RHS;
                         } else {
                             choices[choice_index] = 0;
                             regs[c.out] = LHS.min(RHS);
                         }
                         choice_index++;
                         break;
            case OP_MAX: if (LHS.lower > RHS.upper) {
                             choices[choice_index] = 1;
                             regs[c.out] = LHS;
                         } else if (RHS.lower > LHS.upper) {
                             choices[choice_index] = 2;
                             regs[c.out] = RHS;
                         } else {
                             choices[choice_index] = 0;
                             regs[c.out] = LHS.max(RHS);
                         }
                         choice_index++;
                         break;
            case OP_SUB: regs[c.out] = LHS - RHS; break;

            // Skipping various hard functions here
            default: break;
        }
    }
#undef LHS
#undef RHS
}

__device__
void Renderable::processTiles(const uint32_t offset, const View& v)
{
    // This should be a 1D kernel
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;
    auto regs = regs_i + start * tape.num_regs;

    const uint32_t index = start + offset;
    if (index >= TOTAL_TILES) {
        return;
    }
    const float x = index / TILE_COUNT;
    const float y = index % TILE_COUNT;

    Interval X = {x / TILE_COUNT, (x + 1) / TILE_COUNT};
    X.lower = 2.0f * (X.lower - 0.5f - v.center[0]) * v.scale;
    X.upper = 2.0f * (X.upper - 0.5f - v.center[0]) * v.scale;

    Interval Y = {y / TILE_COUNT, (y + 1) / TILE_COUNT};
    Y.lower = 2.0f * (Y.lower - 0.5f - v.center[1]) * v.scale;
    Y.upper = 2.0f * (Y.upper - 0.5f - v.center[1]) * v.scale;

    // Unpack a 1D offset into the data arrays
    auto csg_choices = this->csg_choices + index * tape.num_csg_choices;
    walkI(tape, X, Y, regs, csg_choices);

    const Interval result = regs[tape[tape.num_clauses - 1].out];
    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper < 0.0f) {
        const uint32_t i = atomicAdd(&filled_tiles, 1);
        tiles[TOTAL_TILES*2 - 1 - i] = index;
    }

    // If the tile is ambiguous, then record it as needing further refinement
    else if (result.lower <= 0.0f && result.upper >= 0.0f) {
        // Store the linked list of subtapes into the active tiles list
        const uint32_t i = atomicAdd(&active_tiles, 1);
        tiles[2 * i] = index;
    }
}

__global__ void processTiles(Renderable* r, const uint32_t offset,
                             Renderable::View v)
{
    r->processTiles(offset, v);
}

__device__
void Renderable::buildSubtapes(const uint32_t offset)
{
    // This is a 1D kernel which consumes tiles and writes out their tapes
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;

    // Reuse the registers array to track activeness
    auto regs = regs_i + start * tape.num_regs;
    const uint32_t i = start + offset;
    if (i >= active_tiles) {
        return;
    }
    const uint32_t index = tiles[2 * i];

    bool* __restrict__ active = reinterpret_cast<bool*>(regs);
    for (uint32_t j=0; j < tape.num_regs; ++j) {
        active[j] = false;
    }

    // Pick an offset CSG choices array
    auto csg_choices = this->csg_choices + index * tape.num_csg_choices;

    // Mark the root of the tree as true
    uint32_t t = tape.num_clauses;
    active[tape[t - 1].out] = true;

    // Begin walking down CSG choices
    uint32_t c = tape.num_csg_choices;

    // Claim a subtape to populate
    uint32_t subtape_index = atomicAdd(&active_subtapes, 1);
    assert(subtape_index < LIBFIVE_CUDA_NUM_SUBTAPES);

    // Since we're reversing the tape, this is going to be the
    // end of the linked list (i.e. next = 0)
    Subtape* subtape = &subtapes[subtape_index];
    subtape->next = 0;
    uint32_t s = 0;

    // Walk from the root of the tape downwards
    while (t--) {
        using namespace libfive::Opcode;
        if (active[tape[t].out]) {
            active[tape[t].out] = false;
            uint32_t mask = 0;
            const uint32_t lhs = (tape[t].banks & 1) ? 0 : tape[t].lhs;
            const uint32_t rhs = (tape[t].banks & 2) ? 0 : tape[t].rhs;
            if (tape[t].opcode == OP_MIN || tape[t].opcode == OP_MAX)
            {
                uint8_t choice = csg_choices[--c];
                if (choice == 1) {
                    active[lhs] = true;
                    if (lhs == tape[t].out) {
                        continue;
                    }
                } else if (choice == 2) {
                    active[rhs] = true;
                    if (rhs == tape[t].out) {
                        continue;
                    }
                } else if (choice == 0) {
                    active[lhs] = true;
                    active[rhs] = true;
                } else {
                    assert(false);
                }
                mask = (choice << 30);
            } else {
                active[lhs] = true;
                active[rhs] = true;
            }

            if (s == LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
                auto next_subtape_index = atomicAdd(&active_subtapes, 1);
                auto next_subtape = &subtapes[next_subtape_index];
                subtape->size = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                next_subtape->next = subtape_index;

                subtape_index = next_subtape_index;
                subtape = next_subtape;
                s = 0;
            }
            (*subtape)[s++] = (t | mask);
        } else if (tape[t].opcode == OP_MIN || tape[t].opcode == OP_MAX) {
            --c;
        }
    }
    // The last subtape may not be completely filled
    subtape->size = s;

    // Store the linked list of subtapes into the active tiles list
    tiles[2 * i + 1] = subtape_index;
}

__global__ void buildSubtapes(Renderable* r, const uint32_t offset) {
    r->buildSubtapes(offset);
}

////////////////////////////////////////////////////////////////////////////////

__global__ void drawFilledTiles(Renderable* r, const uint32_t offset, Renderable::View v) {
    r->drawFilledTiles(offset, v);
}

__device__ void Renderable::drawFilledTiles(const uint32_t offset, const View& v)
{
    // Each thread picks a block and fills in the whole thing
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t i = start + offset;
    if (i >= filled_tiles) {
        return;
    }

    const uint32_t tile = tiles[TOTAL_TILES*2 - i - 1];

    // Convert from tile position to pixels
    const uint32_t px = (tile / TILE_COUNT) * LIBFIVE_CUDA_TILE_SIZE_PX;
    const uint32_t py = (tile % TILE_COUNT) * LIBFIVE_CUDA_TILE_SIZE_PX;

    uint4* pix = reinterpret_cast<uint4*>(&image[px + py * IMAGE_SIZE_PX]);
    const uint4 fill = make_uint4(0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0);
    for (unsigned y=0; y < LIBFIVE_CUDA_TILE_SIZE_PX; y++) {
        for (unsigned x=0; x < LIBFIVE_CUDA_TILE_SIZE_PX; x += 16) {
            *pix = fill;
            pix++;
        }
        pix += (IMAGE_SIZE_PX - LIBFIVE_CUDA_TILE_SIZE_PX) / 16;
    }
}

////////////////////////////////////////////////////////////////////////////////

__device__ float walkF(const Tape& tape,
                       const Subtape* const subtapes,
                       uint32_t subtape_index,
                       const float x, const float y,
                       float* const __restrict__ regs)
{
    assert(subtape_index != 0);
    uint32_t s = subtapes[subtape_index].size;
    uint32_t target;
    while (true) {
        if (s == 0) {
            if (subtapes[subtape_index].next) {
                subtape_index = subtapes[subtape_index].next;
                s = subtapes[subtape_index].size;
            } else {
                return regs[tape[target].out];
            }
        }
        s -= 1;

        // Pick the target, which is an offset into the original tape
        target = subtapes[subtape_index][s];

        // Mask out choice bits
        const uint8_t choice = (target >> 30);
        target &= (1 << 30) - 1;

        const Clause c = tape[target];

#define LHS (!(c.banks & 1) ? regs[c.lhs] : tape.constant(c.lhs))
#define RHS (!(c.banks & 2) ? regs[c.rhs] : tape.constant(c.rhs))
        using namespace libfive::Opcode;
        switch (c.opcode) {
            case VAR_X: regs[c.out] = x; break;
            case VAR_Y: regs[c.out] = y; break;
            case VAR_Z: regs[c.out] = 0.0f; break;

            case OP_SQUARE: regs[c.out] = LHS * LHS; break;
            case OP_SQRT: regs[c.out] = sqrtf(LHS); break;
            case OP_NEG: regs[c.out] = -LHS; break;
            // Skipping transcendental functions for now

            case OP_ADD: regs[c.out] = LHS + RHS; break;
            case OP_MUL: regs[c.out] = LHS * RHS; break;
            case OP_DIV: regs[c.out] = LHS / RHS; break;
            case OP_MIN: if (choice == 1) {
                            regs[c.out] = LHS;
                        } else if (choice == 2) {
                            regs[c.out] = RHS;
                        } else {
                            regs[c.out] = fminf(LHS, RHS);
                        }
                        break;
            case OP_MAX: if (choice == 1) {
                           regs[c.out] = LHS;
                        } else if (choice == 2) {
                           regs[c.out] = RHS;
                        } else {
                           regs[c.out] = fmaxf(LHS, RHS);
                        }
                        break;
            case OP_SUB: regs[c.out] = LHS - RHS; break;

            // Skipping various hard functions here
            default: break;
        }
    }
#undef LHS
#undef RHS
    assert(false);
    return 0.0f;
}

__device__ void Renderable::drawAmbiguousTiles(const uint32_t offset, const View& v)
{
    // We assume one thread per pixel in a tile
    assert(blockDim.x == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(blockDim.x == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;

    // Pick an index into the register array
    const uint32_t pos = (blockIdx.x * LIBFIVE_CUDA_TILE_SIZE_PX + dx) *
                          LIBFIVE_CUDA_TILE_SIZE_PX + dy;
    float* const __restrict__ regs = regs_f + pos * tape.num_regs;

    // Pick an active tile from the list
    const uint32_t i = offset + blockIdx.x;
    if (i >= active_tiles) {
        return;
    }
    const uint32_t tile = tiles[i * 2];
    const uint32_t subtape_index = tiles[i * 2 + 1];

    // Convert from tile position to pixels
    uint32_t px = (tile / TILE_COUNT) * LIBFIVE_CUDA_TILE_SIZE_PX + dx;
    uint32_t py = (tile % TILE_COUNT) * LIBFIVE_CUDA_TILE_SIZE_PX + dy;

    float x = px / (IMAGE_SIZE_PX - 1.0f);
    float y = py / (IMAGE_SIZE_PX - 1.0f);
    x = 2.0f * (x - 0.5f - v.center[0]) * v.scale;
    y = 2.0f * (y - 0.5f - v.center[1]) * v.scale;
    const float f = walkF(tape, subtapes, subtape_index, x, y, regs);
    if (f < 0.0f) {
        image[px + py * IMAGE_SIZE_PX] = 255;
    }
}

__global__ void drawAmbiguousTiles(Renderable* r, const uint32_t offset,
                                   Renderable::View v)
{
    r->drawAmbiguousTiles(offset, v);
}

////////////////////////////////////////////////////////////////////////////////

void Renderable::run(const View& view)
{
    cudaStream_t streams[2] = {this->streams[0], this->streams[1]};

    // Reset our counter variables
    active_tiles = 0;
    filled_tiles = 0;
    active_subtapes = 1;

    // Record this local variable because otherwise it looks up memory
    // that has been loaned to the GPU and not synchronized.
    const uint32_t total_tiles = TOTAL_TILES;
    const uint32_t stride = LIBFIVE_CUDA_TILE_THREADS *
                            LIBFIVE_CUDA_TILE_BLOCKS;

    // Eventually, we'll move the tape data to constant memory
    tape.pointTo(tape.data);

    // Do per-tile evaluation to get filled / ambiguous tiles
    for (unsigned i=0; i < total_tiles; i += stride) {
        ::processTiles<<<LIBFIVE_CUDA_TILE_BLOCKS,
                         LIBFIVE_CUDA_TILE_THREADS,
                         0, streams[0]>>>(this, i, view);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaStreamSynchronize(streams[0]));

    // Pull a few variables back from the GPU
    const uint32_t filled_tiles = this->filled_tiles;
    const uint32_t active_tiles = this->active_tiles;

    for (unsigned i=0; i < filled_tiles; i += stride) {
        // Drawing filled and ambiguous tiles can happen simultaneously,
        // so we assign each one to a separate stream.
        ::drawFilledTiles<<<LIBFIVE_CUDA_TILE_BLOCKS,
                            LIBFIVE_CUDA_TILE_THREADS,
                            0, streams[1]>>>(this, i, view);
        CHECK(cudaGetLastError());
    }

    // Build subtapes in memory for ambiguous tiles
    for (unsigned i=0; i < active_tiles; i += stride) {
        ::buildSubtapes<<<LIBFIVE_CUDA_TILE_BLOCKS,
                          LIBFIVE_CUDA_TILE_THREADS,
                          0, streams[0]>>>(this, i);
        CHECK(cudaGetLastError());
    }

    // Do pixel-by-pixel rendering for ambiguous tiles
    for (unsigned i=0; i < active_tiles; i += LIBFIVE_CUDA_RENDER_BLOCKS) {
        const dim3 T(LIBFIVE_CUDA_TILE_SIZE_PX, LIBFIVE_CUDA_TILE_SIZE_PX);
        ::drawAmbiguousTiles<<<LIBFIVE_CUDA_RENDER_BLOCKS,
                               T, 0, streams[0]>>>(this, i, view);
        CHECK(cudaGetLastError());
    }
}
