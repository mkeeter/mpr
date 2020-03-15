#pragma once

/*  Initial rendering of top-level tiles is done with this many blocks
 *  and threads.  This isn't particularly performance-sensitive; later
 *  operations take much more of our computing time. */
#define LIBFIVE_CUDA_TILE_BLOCKS 16
#define LIBFIVE_CUDA_TILE_THREADS 256

/*  Rendering normals is done with one thread per subtile in the target
 *  tile, with a small number of tiles grouped together into each block
 *  (to avoid running with a very small number of threads) */
#define LIBFIVE_CUDA_REFINE_BLOCKS 512
#define LIBFIVE_CUDA_REFINE_TILES_PER_BLOCK 2

/*  Rendering normals is done with one thread per pixel/voxel in the target
 *  tile, with a small number of tiles grouped together into each block
 *  (to avoid running with a very small number of threads) */
#define LIBFIVE_CUDA_PIXEL_RENDER_BLOCKS 512
#define LIBFIVE_CUDA_PIXEL_RENDER_TILES_PER_BLOCK 2

/*  Rendering normals is done with one thread per pixel in the target
 *  tile, with a small number of tiles grouped together into each block
 *  (to avoid running with a very small number of threads) */
#define LIBFIVE_CUDA_NORMAL_RENDER_BLOCKS 512
#define LIBFIVE_CUDA_NORMAL_RENDER_TILES_PER_BLOCK 2

/*  Split the work among a bunch of streams to maximize utilization */
#define LIBFIVE_CUDA_NUM_STREAMS 2

/*  This is the number of subtapes allocated, where each subtape is a
 *  chunk with some number of clauses. */
#define LIBFIVE_CUDA_NUM_SUBTAPES 320000
#define LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE 64

/*  Depending on compute capabilities, you'll have more or less
 *  shared memory to work with (either 48K or 96K per SM / block).
 *  Set this to apply a global scale to the size of shared caches. */
#define LIBFIVE_CUDA_SM_SCALE 1

//#define USE_AFFINE
