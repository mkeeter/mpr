#pragma once

/*  Defines how many pixels are on the side of each tile and subtile */
#define LIBFIVE_CUDA_TILE_SIZE_PX 64
#define LIBFIVE_CUDA_SUBTILE_SIZE_PX 8
#define LIBFIVE_CUDA_SUBTILES_PER_TILE_SIDE \
    (LIBFIVE_CUDA_TILE_SIZE_PX / LIBFIVE_CUDA_SUBTILE_SIZE_PX)
#define LIBFIVE_CUDA_SUBTILES_PER_TILE \
    (LIBFIVE_CUDA_SUBTILES_PER_TILE_SIDE* LIBFIVE_CUDA_SUBTILES_PER_TILE_SIDE)
#define LIBFIVE_CUDA_PIXELS_PER_SUBTILE \
    (LIBFIVE_CUDA_SUBTILE_SIZE_PX * LIBFIVE_CUDA_SUBTILE_SIZE_PX)

/*  All tile operations are done with this many blocks and threads,
 *  looping to consume all available tiles */
#define LIBFIVE_CUDA_TILE_BLOCKS 16
#define LIBFIVE_CUDA_TILE_THREADS 256

/*  Rendering is done with this many blocks, and SUBTILES_PER_TILE threads
 *  (one per subtile in each tile) */
#define LIBFIVE_CUDA_SUBTILE_BLOCKS 128

/*  Refining subtile tapes is done with this many threads */
#define LIBFIVE_CUDA_SUBTILE_THREADS 256

/*  Rendering is done with this many blocks, and PIXELS_PER_SUBTILE threads
 *  (one per pixel in each tile) */
#define LIBFIVE_CUDA_RENDER_BLOCKS 64

/*  This is the number of subtapes allocated.  Each subtape has room for some
 *  number of clauses, defined in the Subtape struct */
#define LIBFIVE_CUDA_NUM_SUBTAPES 65536

/*  This is the length of each subtape chunk */
#define LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE 256
