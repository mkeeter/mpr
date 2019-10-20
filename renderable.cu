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
          std::max(TOTAL_TILES * (sizeof(Interval) * tape.num_regs +
                                  max(1, tape.num_csg_choices)),
                   sizeof(float) * tape.num_regs * LIBFIVE_CUDA_NUM_FILL_BLOCKS
                                 * LIBFIVE_CUDA_TILE_SIZE_PX
                                 * LIBFIVE_CUDA_TILE_SIZE_PX))),
      regs_i(reinterpret_cast<Interval*>(scratch)),
      csg_choices(scratch + TOTAL_TILES * sizeof(Interval) * tape.num_regs),
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
        // Store the linked list of subtapes into the active tiles list
        uint32_t i = atomicAdd(&active_tiles, 1);
        tiles[2 * i] = index;
    }
}

__global__ void processTiles(Renderable* r, Renderable::View v) {
    r->processTiles(v);
}

__device__
void Renderable::buildSubtapes()
{
    // This is a 1D kernel which consumes tiles and writes out their tapes
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t num_active = active_tiles;
    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t offset = blockDim.x * gridDim.x;
    for (uint32_t i=start; i < num_active; i += offset) {
        const uint32_t index = tiles[2 * i];

        // Reuse the registers array to track activeness
        auto regs = regs_i + index * tape.num_regs;
        bool* __restrict__ active = reinterpret_cast<bool*>(regs);
        for (uint32_t j=0; j < tape.num_regs; ++j) {
            active[i] = false;
        }

        // Pick an offset CSG choices array
        auto csg_choices = this->csg_choices + index * tape.num_csg_choices;

        // Mark the root of the tree as true
        uint32_t t = tape.tape_length;
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
            if (active[tape[t].out]) {
                active[tape[t].out] = false;
                using namespace libfive::Opcode;
                uint32_t mask = 0;
                if (tape[t].opcode == OP_MIN || tape[t].opcode == OP_MAX)
                {
                    uint8_t choice = csg_choices[--c];
                    if (choice == 1) {
                        active[tape[t].lhs] = true;
                    } else if (choice == 2) {
                        active[tape[t].rhs] = true;
                    } else if (choice == 0) {
                        active[tape[t].lhs] = true;
                        active[tape[t].rhs] = true;
                    } else {
                        assert(false);
                    }
                    mask = (choice << 30);
                } else {
                    active[tape[t].lhs] = true;
                    active[tape[t].rhs] = true;
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
            }
        }
        // The last subtape may not be completely filled
        subtape->size = s;

        // Store the linked list of subtapes into the active tiles list
        tiles[2 * i + 1] = subtape_index;
    }
}

__global__ void buildSubtapes(Renderable* r) {
    r->buildSubtapes();
}

////////////////////////////////////////////////////////////////////////////////

__global__ void drawFilledTiles(Renderable* r, Renderable::View v) {
    r->drawFilledTiles(v);
}

__device__ void Renderable::drawFilledTiles(const View& v)
{
    // Each thread picks a block and fills in the whole thing
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t num_filled = filled_tiles;
    const uint32_t start = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t offset = blockDim.x * gridDim.x;
    for (uint32_t i=start; i < num_filled; i += offset) {
        const uint32_t index = tiles[2 * i];
        const uint32_t tile = tiles[TOTAL_TILES*2 - i - 1];

        // Convert from tile position to pixels
        const uint32_t px = (tile / TILE_COUNT) * LIBFIVE_CUDA_TILE_SIZE_PX;
        const uint32_t py = (tile % TILE_COUNT) * LIBFIVE_CUDA_TILE_SIZE_PX;

        uint4* pix = reinterpret_cast<uint4*>(&image[px + py * IMAGE_SIZE_PX]);
        const uint4 fill = make_uint4(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        for (unsigned y=0; y < LIBFIVE_CUDA_TILE_SIZE_PX; y++) {
            for (unsigned x=0; x < LIBFIVE_CUDA_TILE_SIZE_PX; x += 16) {
                *pix = fill;
                pix++;
            }
            pix += (IMAGE_SIZE_PX - LIBFIVE_CUDA_TILE_SIZE_PX) / 16;
        }
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
    assert(blockDim.x == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(blockDim.x == LIBFIVE_CUDA_TILE_SIZE_PX);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;

    // Pick an index into the register array
    uint32_t offset = (blockIdx.x * LIBFIVE_CUDA_TILE_SIZE_PX + dx) *
                      LIBFIVE_CUDA_TILE_SIZE_PX + dy;
    float* const __restrict__ regs = regs_f + offset * tape.num_regs;

    const uint32_t num_active = active_tiles;
    for (uint32_t i=blockIdx.x; i < num_active; i += gridDim.x) {
        // Pick an active tile from the list
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
    const uint32_t N = TILE_COUNT / LIBFIVE_CUDA_THREADS_PER_INTERVAL_BLOCK;
    dim3 grid_i(N, N);
    dim3 threads_i(LIBFIVE_CUDA_THREADS_PER_INTERVAL_BLOCK,
                   LIBFIVE_CUDA_THREADS_PER_INTERVAL_BLOCK);

    dim3 grid_p(LIBFIVE_CUDA_NUM_FILL_BLOCKS);
    dim3 threads_p(LIBFIVE_CUDA_TILE_SIZE_PX, LIBFIVE_CUDA_TILE_SIZE_PX);

    dim3 grid_a(LIBFIVE_CUDA_NUM_AMBIGUOUS_BLOCKS);
    cudaStream_t streams[2] = {this->streams[0], this->streams[1]};

    // Reset our counter variables
    active_tiles = 0;
    filled_tiles = 0;
    active_subtapes = 1;

    ::processTiles<<<grid_i, threads_i, 0, streams[0]>>>(this, view);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(streams[0]));

    // Drawing filled and ambiguous tiles can happen simultaneously,
    // so we assign each one to a separate stream.
    ::drawFilledTiles<<<16, 256, 0, streams[1]>>>(this, view);
    CHECK(cudaGetLastError());

    ::buildSubtapes<<<16, 256, 0, streams[0]>>>(this);
    CHECK(cudaGetLastError());

    ::drawAmbiguousTiles<<<grid_a, threads_p, 0, streams[0]>>>(this, view);
    CHECK(cudaGetLastError());
}
