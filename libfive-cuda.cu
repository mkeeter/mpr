// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <math_constants.h>

// libfive
#include <libfive/tree/opcode.hpp>
#include <libfive/tree/tree.hpp>

// Our Interval arithmetic class
#include "gpu_interval.hpp"
#include "clause.hpp"
#include "check.hpp"
#include "tape.hpp"
#include "subtape.hpp"

__device__ void walkI(const Tape tape,
                      const Interval X, const Interval Y,
                      Interval* const __restrict__ regs,
                      uint8_t* const __restrict__ choices)
{
    uint32_t choice_index = 0;
    for (uint32_t i=0; i < tape.tape_length; ++i) {
        const Clause c = tape[i];
#define LHS ((!(c.banks & 1) ? regs[c.lhs] : Interval{tape.constants[c.lhs], \
                                                     tape.constants[c.lhs]}))
#define RHS ((!(c.banks & 2) ? regs[c.rhs] : Interval{tape.constants[c.rhs], \
                                                     tape.constants[c.rhs]}))
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

__device__ float walkF(const Tape tape,
                      const Subtape* const subtapes,
                      uint32_t subtape_index,
                      const float x, const float y,
                      float* const __restrict__ regs)
{
    assert(subtape_index != 0);
    uint32_t s = subtapes[subtape_index].size;
    while (true) {
        if (s == 0) {
            if (subtapes[subtape_index].next) {
                subtape_index = subtapes[subtape_index].next;
                s = subtapes[subtape_index].size;
            } else {
                return regs[tape[subtapes[subtape_index].subtape[0]].out];
            }
        }
        s -= 1;

        // Mask out choice bits
        const uint8_t choice = (s >> 30);
        s &= (1 << 30) - 1;

        const Clause c = tape[subtapes[subtape_index].subtape[s]];

#define LHS (!(c.banks & 1) ? regs[c.lhs] : tape.constants[c.lhs])
#define RHS (!(c.banks & 2) ? regs[c.rhs] : tape.constants[c.rhs])
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

struct Output {
    uint32_t* const __restrict__ tiles;
    const uint32_t tiles_length;

    uint32_t num_active;
    uint32_t num_filled;

    Subtape* const __restrict__ subtapes;
    const uint32_t subtapes_length;
    uint32_t num_subtapes;
};

__global__ void processTiles(const Tape tape,
        // Flat array for all pseudoregisters
        Interval* const __restrict__ regs_,

        // Flat array for all CSG choices
        uint8_t* const __restrict__ csg_choices_,

        // Output data
        Output* const __restrict__ out)
{
    assert(blockDim.x == blockDim.y);
    assert(gridDim.x == gridDim.y);

    const float x = blockIdx.x * blockDim.x + threadIdx.x;
    const float y = blockIdx.y * blockDim.y + threadIdx.y;

    const uint32_t TILE_COUNT = gridDim.x * blockDim.x;

    const Interval X = {x / TILE_COUNT, (x + 1) / TILE_COUNT};
    const Interval Y = {y / TILE_COUNT, (y + 1) / TILE_COUNT};

    // Unpack a 1D offset into the data arrays
    const uint32_t index = x * TILE_COUNT + y;
    auto regs = regs_ + index * tape.num_regs;
    auto csg_choices = csg_choices_ + index * tape.num_csg_choices;
    walkI(tape, X, Y, regs, csg_choices);

    const Interval result = regs[tape[tape.tape_length - 1].out];
    // If this tile is unambiguously filled, then mark it at the end
    // of the tiles list
    if (result.upper < 0.0f) {
        uint32_t i = atomicAdd(&out->num_filled, 1);
        out->tiles[out->tiles_length - 1 - i] = index;
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
        uint32_t subtape_index = atomicAdd(&out->num_subtapes, 1);
        assert(subtape_index < out->subtapes_length);

        // Since we're reversing the tape, this is going to be the
        // end of the linked list (i.e. next = 0)
        Subtape* subtape = &out->subtapes[subtape_index];
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
                    }
                    mask = (choice << 30);
                } else {
                    active[tape[t].lhs] = true;
                    active[tape[t].rhs] = true;
                }

                if (s == SUBTAPE_LENGTH) {
                    auto next_subtape_index = atomicAdd(&out->num_subtapes, 1);
                    auto next_subtape = &out->subtapes[next_subtape_index];
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
        uint32_t i = atomicAdd(&out->num_active, 1);
        out->tiles[2 * i] = index;
        out->tiles[2 * i + 1] = subtape_index;
    }
}

template <unsigned TILE_COUNT>
__global__ void fillTiles(Output* const __restrict__ out,
                          uint8_t* __restrict__ image)
{
    // We assume one thread per pixel in a tile
    const uint32_t TILE_SIZE_PX = blockDim.x;
    assert(blockDim.x == blockDim.y);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;

    const uint32_t num_filled = out->num_filled;
    for (uint32_t i=blockIdx.x; i < num_filled; i += gridDim.x) {
        // Pick a filled tile from the list
        const uint32_t tile = out->tiles[out->tiles_length - i - 1];

        // Convert from tile position to pixels
        const uint32_t px = (tile / TILE_COUNT) * TILE_SIZE_PX + dx;
        const uint32_t py = (tile % TILE_COUNT) * TILE_SIZE_PX + dy;

        image[px + py * TILE_SIZE_PX * TILE_COUNT] = 1;
    }
}

template <unsigned TILE_COUNT>
__global__ void renderTiles(const Tape tape,
                            const Output* const __restrict__ out,

                            // Flat array for all pseudoregisters
                            float* const __restrict__ regs_,

                            uint8_t* __restrict__ image)
{
    // We assume one thread per pixel in a tile
    const uint32_t TILE_SIZE_PX = blockDim.x;
    assert(blockDim.x == blockDim.y);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    const uint32_t dx = threadIdx.x;
    const uint32_t dy = threadIdx.y;

    // Pick an index into the register array
    uint32_t offset = (blockIdx.x * TILE_SIZE_PX + dx) * TILE_SIZE_PX + dy;
    float* const __restrict__ regs = regs_ + offset * tape.num_regs;

    const uint32_t num_active = out->num_active;
    for (uint32_t i=blockIdx.x; i < num_active; i += gridDim.x) {
        // Pick an active tile from the list
        const uint32_t tile = out->tiles[i * 2];
        const uint32_t subtape_index = out->tiles[i * 2 + 1];

        // Convert from tile position to pixels
        const uint32_t px = (tile / TILE_COUNT) * TILE_SIZE_PX + dx;
        const uint32_t py = (tile % TILE_COUNT) * TILE_SIZE_PX + dy;

        const float x = px / (TILE_SIZE_PX * TILE_COUNT - 1.0f);
        const float y = py / (TILE_SIZE_PX * TILE_COUNT - 1.0f);
        const float f = walkF(tape, out->subtapes, subtape_index, x, y, regs);

        image[px + py * TILE_SIZE_PX * TILE_COUNT] = (f < 0.0f) ? 255 : 0;
    }
}

template <unsigned IMAGE_SIZE_PX=4096, unsigned TILE_SIZE_PX=16>
Output* callProcessTiles(Tape tape) {
    constexpr unsigned TILE_COUNT = IMAGE_SIZE_PX / TILE_SIZE_PX;
    constexpr unsigned TOTAL_TILES = TILE_COUNT * TILE_COUNT;

    constexpr unsigned NUM_BLOCKS = 8;
    constexpr unsigned THREADS_PER_BLOCK = TILE_COUNT / NUM_BLOCKS;

    const unsigned FILL_BLOCKS = 1024;
    printf("threads per block: %u\n", THREADS_PER_BLOCK);

    Interval* d_regs_i;
    CHECK(cudaMallocManaged(
          reinterpret_cast<void **>(&d_regs_i),
          sizeof(Interval) * tape.num_regs * TOTAL_TILES));

    float* d_regs_f;
    CHECK(cudaMallocManaged(
          reinterpret_cast<void **>(&d_regs_f),
          sizeof(float) * tape.num_regs * FILL_BLOCKS
                        * TILE_SIZE_PX * TILE_SIZE_PX));

    uint8_t* d_csg_choices;
    CHECK(cudaMallocManaged(
          reinterpret_cast<void **>(&d_csg_choices),
          max(1, tape.num_csg_choices) * TOTAL_TILES));

    uint32_t* d_tiles;
    CHECK(cudaMallocManaged(
          reinterpret_cast<void **>(&d_tiles),
          sizeof(uint32_t) * 2 * TOTAL_TILES));

    Output* d_out;
    CHECK(cudaMallocManaged(
          reinterpret_cast<void **>(&d_out),
          sizeof(Output)));

    Subtape* d_subtapes;
    const static uint32_t subtapes_length = 65535;
    CHECK(cudaMallocManaged(
          reinterpret_cast<void **>(&d_subtapes),
          sizeof(Subtape) * subtapes_length));

    CHECK(cudaDeviceSynchronize());
    new (d_out) Output { d_tiles, TOTAL_TILES * 2,
        0, /* num_active */
        0, /* num_filled */
        d_subtapes,
        subtapes_length,
        1 /* We start at subtape 1, to use 0 as a list terminator */
    };

    {
        dim3 grid(NUM_BLOCKS, NUM_BLOCKS);
        dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        printf("threads per block: %u\tnumber of blocks: %u\n",
                THREADS_PER_BLOCK, NUM_BLOCKS);

        processTiles <<< grid, threads >>>(tape, d_regs_i, d_csg_choices, d_out);
        const auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            fprintf(stderr, "Failed to launch: %s\n",
                    cudaGetErrorString(code));
        }
    }

    {
        dim3 threads(TILE_SIZE_PX, TILE_SIZE_PX);

        uint8_t* d_image;
        CHECK(cudaMallocManaged(
              (void**)&d_image, IMAGE_SIZE_PX * IMAGE_SIZE_PX));
        CHECK(cudaDeviceSynchronize());
        cudaMemset(d_image, 0, IMAGE_SIZE_PX * IMAGE_SIZE_PX);

        fillTiles<TILE_COUNT> <<< FILL_BLOCKS, threads >>>(d_out, d_image);
        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            fprintf(stderr, "Failed to launch: %s\n",
                    cudaGetErrorString(code));
        }

        renderTiles<TILE_COUNT> <<< FILL_BLOCKS, threads >>>(tape, d_out,
                d_regs_f, d_image);
        code = cudaGetLastError();
        if (code != cudaSuccess) {
            fprintf(stderr, "Failed to launch: %s\n",
                    cudaGetErrorString(code));
        }

        CHECK(cudaDeviceSynchronize());
        printf("Got %u subtapes\n", d_out->num_subtapes);
        printf("subtape 1 next: %u\n", d_out->subtapes[1].next);
        printf("subtape 1 size: %u\n", d_out->subtapes[1].size);
        printf("subtape 1 values:\n");
        for (unsigned i=0; i < d_out->subtapes[1].size; ++i) {
            printf("%u ", d_out->subtapes[1].subtape[i]);
        }
        printf("\n");

#if 0
        for (unsigned i=0; i < IMAGE_SIZE_PX * IMAGE_SIZE_PX; ++i) {
            if (i && !(i % IMAGE_SIZE_PX)) {
                printf("\n");
            }
            const char c = d_image[i] ? ('0' + (i%10)) : ' ';
            printf("%c", c);
        }
        printf("\n");
#endif
    }
    return d_out;
}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto circle = sqrt(X*X + Y*Y) - 1.0;
    auto tape = Tape::build(circle);

    auto d_out = callProcessTiles(tape);
    cudaDeviceSynchronize();
    printf("%u %u\n", d_out->num_active, d_out->num_filled);

    return 0;
}
