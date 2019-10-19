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

#if 0
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
#endif

/**
 * Program main
 */
int main(int argc, char **argv)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto circle = sqrt(X*X + Y*Y) - 1.0;
    auto tape = Tape::build(circle);

    //auto d_out = callProcessTiles(tape);
    cudaDeviceSynchronize();
    //printf("%u %u\n", d_out->num_active, d_out->num_filled);

    return 0;
}
