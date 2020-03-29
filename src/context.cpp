/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include "context.hpp"
#include "parameters.hpp"

namespace libfive {
namespace cuda {

Context::Context(int32_t image_size_px)
    : image_size_px(image_size_px)
{
    // Build the four stages
    for (unsigned i=0; i < 4; ++i) {
        const unsigned tile_size_px = 64 / (1 << (i * 2));
        stages[i].filled.reset(CUDA_MALLOC(
                int32_t,
                pow(image_size_px / tile_size_px, 2)));
    }

    normals.reset(CUDA_MALLOC(uint32_t, image_size_px * image_size_px));

    // Allocate a bunch of memory to store tapes
    tape_data.reset(CUDA_MALLOC(uint64_t, NUM_SUBTAPES * SUBTAPE_CHUNK_SIZE));
    tape_index.reset(CUDA_MALLOC(int32_t, 1));
    *tape_index = 0;

    // Allocate an index to keep track of active tiles
    num_active_tiles.reset(CUDA_MALLOC(int32_t, 1));

    // The first array of tiles must have enough space to hold all of the
    // 64^3 tiles in the volume, which shouldn't be too much.
    stages[0].tiles.reset(CUDA_MALLOC(
            TileNode,
            pow(image_size_px / 64, 3)));

    // We leave the other stage_t's tile arrays unallocated for now, since
    // they're initialized to all zeros and will be resized to fit later.

    // Prefer the L1 cache!
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

} // namespace cuda
} // namespace libfive
