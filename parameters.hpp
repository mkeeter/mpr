#pragma once

/*  Defines how many pixels are on the side of each tile */
#define LIBFIVE_CUDA_TILE_SIZE_PX 16

/*  All tile operations are done with this many blocks and threads,
 *  looping to consume all available tiles */
#define LIBFIVE_CUDA_TILE_BLOCKS 16
#define LIBFIVE_CUDA_TILE_THREADS 256

/*  Rendering is done with this many blogs, and TILE_SIZE_PX^2 threads
 *  (one per pixel in each tile) */
#define LIBFIVE_CUDA_RENDER_BLOCKS 32

/*  This is the number of subtapes allocated.  Each subtape has room for some
 *  number of clauses, defined in the Subtape struct */
#define LIBFIVE_CUDA_NUM_SUBTAPES 65535

/*  This is the length of each subtape chunk */
#define LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE 250
