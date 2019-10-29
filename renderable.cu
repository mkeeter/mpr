#include "check.hpp"
#include "renderable.hpp"
#include "gpu_interval.hpp"
#include "parameters.hpp"

__constant__ static uint64_t const_buffer[0x2000];

void Renderable::Deleter::operator()(Renderable* r)
{
    r->~Renderable();
    CHECK(cudaFree(r));
}

Renderable::~Renderable()
{
    CHECK(cudaFree(scratch));
    CHECK(cudaFree(csg_choices));
    CHECK(cudaFree(tiles));
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

      scratch(CUDA_MALLOC(uint8_t, std::max(floatRegSize(tape.num_regs),
                                            intervalRegSize(tape.num_regs)))),
      regs_i(reinterpret_cast<IntervalRegisters*>(scratch)),
      regs_f(reinterpret_cast<FloatRegisters*>(scratch)),

      csg_choices(CUDA_MALLOC(ChoiceArray, std::max(1U,
                  LIBFIVE_CUDA_TILE_BLOCKS * tape.num_csg_choices *
                  ((TOTAL_TILES + LIBFIVE_CUDA_TILE_THREADS *
                                  LIBFIVE_CUDA_TILE_BLOCKS - 1) /
                   (LIBFIVE_CUDA_TILE_THREADS * LIBFIVE_CUDA_TILE_BLOCKS))))),

      tiles(CUDA_MALLOC(uint32_t, 2 * TOTAL_TILES)),
      active_tiles(0),
      filled_tiles(0),

      active_subtapes(1),

      image(CUDA_MALLOC(uint8_t, IMAGE_SIZE_PX * IMAGE_SIZE_PX))
{
    cudaMemset(image, 0, IMAGE_SIZE_PX * IMAGE_SIZE_PX);
    CHECK(cudaStreamCreate(&streams[0]));
    CHECK(cudaStreamCreate(&streams[1]));
}

////////////////////////////////////////////////////////////////////////////////

__device__ Interval walkI(const Tape& tape,
                      Renderable::IntervalRegisters* const __restrict__ regs,
                      Renderable::ChoiceArray* const __restrict__ choices)
{
    using namespace libfive::Opcode;

    uint32_t choice_index = 0;

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);
    const uint32_t num_clauses = tape.num_clauses;

    for (uint32_t i=0; i < num_clauses; ++i) {
        const Clause c = clause_ptr[i];
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

    const uint32_t index = start + offset;
    if (index >= TOTAL_TILES) {
        return;
    }
    auto regs = regs_i + tape.num_regs * blockIdx.x;
    {   // Prepopulate axis values
        const float x = index / TILE_COUNT;
        const float y = index % TILE_COUNT;

        Interval vs[3];
        const Interval X = {x / TILE_COUNT, (x + 1) / TILE_COUNT};
        vs[0].lower = 2.0f * (X.lower - 0.5f - v.center[0]) * v.scale;
        vs[0].upper = 2.0f * (X.upper - 0.5f - v.center[0]) * v.scale;

        const Interval Y = {y / TILE_COUNT, (y + 1) / TILE_COUNT};
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
    auto csg_choices = this->csg_choices + index / LIBFIVE_CUDA_TILE_THREADS
                                                 * tape.num_csg_choices;

    // Run actual evaluation
    const Interval result = walkI(tape, regs, csg_choices);

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
    const uint32_t i = start + offset;
    if (i >= active_tiles) {
        return;
    }
    const uint32_t index = tiles[2 * i];

    uint8_t* __restrict__ active = scratch + start * tape.num_regs;
    for (uint32_t j=0; j < tape.num_regs; ++j) {
        active[j] = false;
    }

    // Pick an offset CSG choices array
    auto csg_choices = this->csg_choices + index / LIBFIVE_CUDA_TILE_THREADS
                                                 * tape.num_csg_choices;
    const uint32_t j = index % LIBFIVE_CUDA_TILE_THREADS;

    // Mark the root of the tree as true
    uint32_t t = tape.num_clauses;
    active[tape[t - 1].out] = true;

    // Begin walking down CSG choices
    uint32_t csg_choice = tape.num_csg_choices;

    // Claim a subtape to populate
    uint32_t subtape_index = atomicAdd(&active_subtapes, 1);
    assert(subtape_index < LIBFIVE_CUDA_NUM_SUBTAPES);

    // Since we're reversing the tape, this is going to be the
    // end of the linked list (i.e. next = 0)
    subtapes.next[subtape_index] = 0;
    uint32_t s = 0;

    // Walk from the root of the tape downwards
    while (t--) {
        using namespace libfive::Opcode;
        const Clause c = tape[t];
        if (active[c.out]) {
            active[c.out] = false;
            uint32_t mask = 0;
            if (c.opcode == OP_MIN || c.opcode == OP_MAX)
            {
                const uint8_t choice = csg_choices[--csg_choice][j];
                if (choice == 1) {
                    if (!(c.banks & 1)) {
                        active[c.lhs] = true;
                        if (c.lhs == c.out) {
                            continue;
                        }
                    }
                } else if (choice == 2) {
                    active[c.rhs] = true;
                    if (!(c.banks & 2)) {
                        if (c.rhs == c.out) {
                            continue;
                        }
                    }
                } else if (choice == 0) {
                    if (!(c.banks & 1)) {
                        active[c.lhs] = true;
                    }
                    if (!(c.banks & 2)) {
                        active[c.rhs] = true;
                    }
                } else {
                    assert(false);
                }
                mask = (choice << 30);
            } else {
                if (c.opcode >= OP_SQUARE && !(c.banks & 1)) {
                    active[c.lhs] = true;
                }
                if (c.opcode >= OP_ADD && !(c.banks & 2)) {
                    active[c.rhs] = true;
                }
            }

            if (s == LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE) {
                auto next_subtape_index = atomicAdd(&active_subtapes, 1);
                subtapes.size[subtape_index] = LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE;
                subtapes.next[next_subtape_index] = subtape_index;

                subtape_index = next_subtape_index;
                s = 0;
            }
            subtapes.data[subtape_index][s++] = (t | mask);
        } else if (c.opcode == OP_MIN || c.opcode == OP_MAX) {
            --csg_choice;
        }
    }
    // The last subtape may not be completely filled
    subtapes.size[subtape_index] = s;

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
                       const Renderable::Subtapes& subtapes,
                       uint32_t subtape_index,
                       Renderable::FloatRegisters* const __restrict__ regs)
{
    assert(subtape_index != 0);
    using namespace libfive::Opcode;

    const Clause* __restrict__ clause_ptr = &tape[0];
    const float* __restrict__ constant_ptr = &tape.constant(0);

    const uint32_t q = threadIdx.x + threadIdx.y * LIBFIVE_CUDA_TILE_SIZE_PX;
    uint32_t s = subtapes.size[subtape_index];
    uint32_t target;
    while (true) {
        if (s == 0) {
            const uint32_t next = subtapes.next[subtape_index];
            if (next) {
                subtape_index = next;
                s = subtapes.size[subtape_index];
            } else {
                return regs[clause_ptr[target].out][q];
            }
        }
        s -= 1;

        // Pick the target, which is an offset into the original tape
        target = subtapes.data[subtape_index][s];

        // Mask out choice bits
        const uint8_t choice = (target >> 30);
        target &= (1 << 30) - 1;

        const Clause c = clause_ptr[target];

        // All clauses must have at least one argument, since constants
        // and VAR_X/Y/Z are handled separately.
        float lhs;
        if (c.banks & 1) {
            lhs = constant_ptr[c.lhs];
        } else {
            lhs = regs[c.lhs][q];
        }

        float rhs;
        if (c.opcode >= OP_ADD) {
            if (c.banks & 2) {
                rhs = constant_ptr[c.rhs];
            } else {
                rhs = regs[c.rhs][q];
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
            case OP_MIN: if (choice == 1) {
                            out = lhs;
                        } else if (choice == 2) {
                            out = rhs;
                        } else {
                            out = fminf(lhs, rhs);
                        }
                        break;
            case OP_MAX: if (choice == 1) {
                           out = lhs;
                        } else if (choice == 2) {
                           out = rhs;
                        } else {
                           out = fmaxf(lhs, rhs);
                        }
                        break;
            case OP_SUB: out = lhs - rhs; break;

            // Skipping various hard functions here
            default: break;
        }
        regs[c.out][q] = out;
    }
    assert(false);
    return 0.0f;
}

__device__ void Renderable::drawAmbiguousTiles(const uint32_t offset, const View& v)
{
    // We assume one thread per pixel in a tile
    assert(blockDim.x == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(blockDim.y == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;
    const uint32_t q = dx + dy * LIBFIVE_CUDA_TILE_SIZE_PX;

    // Pick an index into the register array
    auto regs = regs_f + tape.num_regs * blockIdx.x;

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

    {   // Prepopulate axis values
        const float x = px / (IMAGE_SIZE_PX - 1.0f);
        const float y = py / (IMAGE_SIZE_PX - 1.0f);
        float vs[3];
        vs[0] = 2.0f * (x - 0.5f - v.center[0]) * v.scale;
        vs[1] = 2.0f * (y - 0.5f - v.center[1]) * v.scale;
        vs[2] = 0.0f;
        for (unsigned i=0; i < 3; ++i) {
            if (tape.axes.reg[i] != UINT16_MAX) {
                regs[tape.axes.reg[i]][q] = vs[i];
            }
        }
    }
    const float f = walkF(tape, subtapes, subtape_index, regs);
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

    tape.sendToConstantMemory((const char*)const_buffer);

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

size_t Renderable::intervalRegSize(uint16_t num_regs) {
    return LIBFIVE_CUDA_TILE_BLOCKS * LIBFIVE_CUDA_TILE_THREADS *
           sizeof(Interval) * num_regs;
}

size_t Renderable::floatRegSize(uint16_t num_regs) {
    return sizeof(float) * num_regs * LIBFIVE_CUDA_RENDER_BLOCKS
                                    * LIBFIVE_CUDA_TILE_SIZE_PX
                                    * LIBFIVE_CUDA_TILE_SIZE_PX;
}
