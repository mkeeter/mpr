#include "check.hpp"
#include "renderable.hpp"
#include "gpu_interval.hpp"

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

std::unique_ptr<Renderable, Renderable::Deleter> Renderable::build(
            libfive::Tree tree,
            uint32_t image_size_px, uint32_t tile_size_px,
            uint32_t num_interval_blocks, uint32_t num_fill_blocks,
            uint32_t num_subtapes)
{
    auto out = CUDA_MALLOC(Renderable, 1);
    new (out) Renderable(tree,
            image_size_px, tile_size_px,
            num_interval_blocks, num_fill_blocks, num_subtapes);
    return std::unique_ptr<Renderable, Deleter>(out);
}

Renderable::Renderable(libfive::Tree tree,
            uint32_t image_size_px, uint32_t tile_size_px,
            uint32_t num_interval_blocks, uint32_t num_fill_blocks,
            uint32_t num_subtapes)
    : tape(std::move(Tape::build(tree))),

      IMAGE_SIZE_PX(image_size_px),
      TILE_SIZE_PX(tile_size_px),
      TILE_COUNT(IMAGE_SIZE_PX / TILE_SIZE_PX),
      TOTAL_TILES(TILE_COUNT * TILE_COUNT),

      NUM_INTERVAL_BLOCKS(num_interval_blocks),
      THREADS_PER_INTERVAL_BLOCK(TILE_COUNT / NUM_INTERVAL_BLOCKS),

      NUM_FILL_BLOCKS(num_fill_blocks),
      NUM_SUBTAPES(num_subtapes),

      scratch(CUDA_MALLOC(uint8_t,
          std::max(TOTAL_TILES * (sizeof(Interval) * tape.num_regs +
                                  max(1, tape.num_csg_choices)),
                   sizeof(float) * tape.num_regs * NUM_FILL_BLOCKS *
                       TILE_SIZE_PX * TILE_SIZE_PX))),
      regs_i(reinterpret_cast<Interval*>(scratch)),
      csg_choices(scratch + TOTAL_TILES * sizeof(Interval) * tape.num_regs),
      regs_f(reinterpret_cast<float*>(scratch)),

      tiles(CUDA_MALLOC(uint32_t, 2 * TOTAL_TILES)),
      active_tiles(0),
      filled_tiles(0),

      subtapes(CUDA_MALLOC(Subtape, NUM_SUBTAPES)),
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
    for (uint32_t i=0; i < tape.tape_length; ++i) {
        const Clause c = tape[i];
#define LHS ((!(c.banks & 1) ? regs[c.lhs] : Interval{tape.constant(c.lhs), \
                                                      tape.constant(c.lhs)}))
#define RHS ((!(c.banks & 2) ? regs[c.rhs] : Interval{tape.constant(c.rhs), \
                                                      tape.constant(c.rhs)}))
        using namespace libfive::Opcode;
        switch (c.opcode) {
            case VAR_X: regs[c.out] = X; break;
            case VAR_Y: regs[c.out] = Y; break;

            case OP_SQUARE: regs[c.out] = LHS.square(); break;
            case OP_SQRT: regs[c.out] = LHS.sqrt(); break;
            case OP_NEG: regs[c.out] = -LHS; break;
            // Skipping transcendental functions for now

            case OP_ADD: regs[c.out] = LHS + RHS; break;
            case OP_MUL: regs[c.out] = LHS * RHS; break;
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
void Renderable::processTiles(const View& v)
{
    assert(blockDim.x == blockDim.y);
    assert(gridDim.x == gridDim.y);

    const float x = blockIdx.x * blockDim.x + threadIdx.x;
    const float y = blockIdx.y * blockDim.y + threadIdx.y;

    Interval X = {x / TILE_COUNT, (x + 1) / TILE_COUNT};
    X.lower = 2.0f * (X.lower - 0.5f - v.center[0]) * v.scale;
    X.upper = 2.0f * (X.upper - 0.5f - v.center[0]) * v.scale;

    Interval Y = {y / TILE_COUNT, (y + 1) / TILE_COUNT};
    Y.lower = 2.0f * (Y.lower - 0.5f - v.center[1]) * v.scale;
    Y.upper = 2.0f * (Y.upper - 0.5f - v.center[1]) * v.scale;

    // Unpack a 1D offset into the data arrays
    const uint32_t index = x * TILE_COUNT + y;
    auto regs = regs_i + index * tape.num_regs;
    auto csg_choices = this->csg_choices + index * tape.num_csg_choices;
    walkI(tape, X, Y, regs, csg_choices);

    const Interval result = regs[tape[tape.tape_length - 1].out];
    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper < 0.0f) {
        uint32_t i = atomicAdd(&filled_tiles, 1);
        tiles[TOTAL_TILES*2 - 1 - i] = index;
    }

    // If the tile is ambiguous, then record it as needing further refinement
    else if (result.lower <= 0.0f && result.upper >= 0.0f) {
        // Reuse the registers array to track activeness
        bool* __restrict__ active = reinterpret_cast<bool*>(regs);
        for (uint32_t i=0; i < tape.num_regs; ++i) {
            active[i] = false;
        }
        // Mark the root of the tree as true
        uint32_t t = tape.tape_length;
        active[tape[t - 1].out] = true;

        // Begin walking down CSG choices
        uint32_t c = tape.num_csg_choices;

        // Claim a subtape to populate
        uint32_t subtape_index = atomicAdd(&active_subtapes, 1);
        assert(subtape_index < NUM_SUBTAPES);

        // Since we're reversing the tape, this is going to be the
        // end of the linked list (i.e. next = 0)
        Subtape* subtape = &subtapes[subtape_index];
        subtape->next = 0;
        const uint32_t SUBTAPE_LENGTH = sizeof( subtape->subtape) /
                                        sizeof(*subtape->subtape);
        uint32_t s = 0;

        // Walk from the root of the tape downwards
        while (t--) {
            if (active[tape[t].out]) {
                using namespace libfive::Opcode;
                uint32_t mask = 0;
                if (tape[t].opcode == OP_MIN || tape[t].opcode == OP_MAX)
                {
                    uint8_t choice = csg_choices[--c];
                    if (choice == 1) {
                        active[tape[t].lhs] = true;
                        active[tape[t].rhs] = false;
                    } else if (choice == 2) {
                        active[tape[t].lhs] = false;
                        active[tape[t].rhs] = true;
                    } else {
                        active[tape[t].lhs] = true;
                        active[tape[t].rhs] = true;
                    }
                    mask = (choice << 30);
                } else {
                    active[tape[t].lhs] = true;
                    active[tape[t].rhs] = true;
                }

                if (s == SUBTAPE_LENGTH) {
                    auto next_subtape_index = atomicAdd(&active_subtapes, 1);
                    auto next_subtape = &subtapes[next_subtape_index];
                    subtape->size = SUBTAPE_LENGTH;
                    next_subtape->next = subtape_index;

                    subtape_index = next_subtape_index;
                    subtape = next_subtape;
                    s = 0;
                }
                subtape->subtape[s++] = (t | mask);
            }
        }
        // The last subtape may not be completely filled
        subtape->size = s;

        // Store the linked list of subtapes into the active tiles list
        uint32_t i = atomicAdd(&active_tiles, 1);
        tiles[2 * i] = index;
        tiles[2 * i + 1] = subtape_index;
    }
}

__global__ void processTiles(Renderable* r, Renderable::View v) {
    r->processTiles(v);
}

////////////////////////////////////////////////////////////////////////////////

__global__ void drawFilledTiles(Renderable* r, Renderable::View v) {
    r->drawFilledTiles(v);
}

__device__ void Renderable::drawFilledTiles(const View& v)
{
    // We assume one thread per pixel in a tile
    assert(blockDim.x == TILE_SIZE_PX);
    assert(blockDim.x == blockDim.y);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;

    const uint32_t num_filled = filled_tiles;
    for (uint32_t i=blockIdx.x; i < num_filled; i += gridDim.x) {
        // Pick a filled tile from the list
        const uint32_t tile = tiles[TOTAL_TILES*2 - i - 1];

        // Convert from tile position to pixels
        const uint32_t px = (tile / TILE_COUNT) * TILE_SIZE_PX + dx;
        const uint32_t py = (tile % TILE_COUNT) * TILE_SIZE_PX + dy;

        image[px + py * TILE_SIZE_PX * TILE_COUNT] = 1;
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
        target = subtapes[subtape_index].subtape[s];

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

            case OP_SQUARE: regs[c.out] = LHS * LHS; break;
            case OP_SQRT: regs[c.out] = sqrtf(LHS); break;
            case OP_NEG: regs[c.out] = -LHS; break;
            // Skipping transcendental functions for now

            case OP_ADD: regs[c.out] = LHS + RHS; break;
            case OP_MUL: regs[c.out] = LHS * RHS; break;
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

__device__ void Renderable::drawAmbiguousTiles(const View& v)
{
    // We assume one thread per pixel in a tile
    assert(blockDim.x == TILE_SIZE_PX);
    assert(blockDim.x == blockDim.y);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;

    // Pick an index into the register array
    uint32_t offset = (blockIdx.x * TILE_SIZE_PX + dx) * TILE_SIZE_PX + dy;
    float* const __restrict__ regs = regs_f + offset * tape.num_regs;

    const uint32_t num_active = active_tiles;
    for (uint32_t i=blockIdx.x; i < num_active; i += gridDim.x) {
        // Pick an active tile from the list
        const uint32_t tile = tiles[i * 2];
        const uint32_t subtape_index = tiles[i * 2 + 1];

        // Convert from tile position to pixels
        const uint32_t px = (tile / TILE_COUNT) * TILE_SIZE_PX + dx;
        const uint32_t py = (tile % TILE_COUNT) * TILE_SIZE_PX + dy;

        float x = px / (TILE_SIZE_PX * TILE_COUNT - 1.0f);
        float y = py / (TILE_SIZE_PX * TILE_COUNT - 1.0f);
        x = 2.0f * (x - 0.5f - v.center[0]) * v.scale;
        y = 2.0f * (y - 0.5f - v.center[1]) * v.scale;
        const float f = walkF(tape, subtapes, subtape_index, x, y, regs);

        image[px + py * TILE_SIZE_PX * TILE_COUNT] = (f < 0.0f) ? 255 : 0;
    }
}

__global__ void drawAmbiguousTiles(Renderable* r, Renderable::View v) {
    r->drawAmbiguousTiles(v);
}

////////////////////////////////////////////////////////////////////////////////

void Renderable::run(const View& view)
{
    // We construct all of these variables first, because the 'this' pointer
    // is allocated in unified memory, so we can't use it after starting
    // kernels (until we call cudaDeviceSynchronize).
    dim3 grid_i(NUM_INTERVAL_BLOCKS, NUM_INTERVAL_BLOCKS);
    dim3 threads_i(THREADS_PER_INTERVAL_BLOCK, THREADS_PER_INTERVAL_BLOCK);

    dim3 grid_p(NUM_FILL_BLOCKS);
    dim3 threads_p(TILE_SIZE_PX, TILE_SIZE_PX);

    cudaStream_t streams[2] = {this->streams[0], this->streams[1]};

    ::processTiles<<<grid_i, threads_i, 0, streams[0]>>>(this, view);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(streams[0]));

    // Drawing filled and ambiguous tiles can happen simultaneously,
    // so we assign each one to a separate stream.
    ::drawFilledTiles<<<grid_p, threads_p, 0, streams[0]>>>(this, view);
    CHECK(cudaGetLastError());

    ::drawAmbiguousTiles<<<grid_p, threads_p, 0, streams[1]>>>(this, view);
    CHECK(cudaGetLastError());
}
