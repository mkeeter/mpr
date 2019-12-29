#pragma once

/*  All tile operations are done with this many blocks and threads,
 *  looping to consume all available tiles */
#define LIBFIVE_CUDA_TILE_BLOCKS 16
#define LIBFIVE_CUDA_TILE_THREADS 256

/*  Refining is done with this many blocks */
#define LIBFIVE_CUDA_SUBTILE_BLOCKS 512
/*  Number of tiles to refine simultaneously */
#define LIBFIVE_CUDA_REFINE_TILES 4

/*  This is the number of subtapes allocated.  Each subtape has room for some
 *  number of clauses, defined in the Subtape struct */
#define LIBFIVE_CUDA_NUM_SUBTAPES (800000*4)
/*  This is the length of each subtape chunk */
#define LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE 64

/*  Rendering is done with this many blocks */
#define LIBFIVE_CUDA_RENDER_BLOCKS 512
/*  Number of subtiles per thread block when rendering */
#define LIBFIVE_CUDA_RENDER_SUBTILES 16

/*  Split the work among a bunch of streams to maximize utilization */
#define LIBFIVE_CUDA_NUM_STREAMS 2

/*  Generating normals is done with this many threads + blocks */
#define LIBFIVE_CUDA_NORMAL_TILES 4
#define LIBFIVE_CUDA_NORMAL_BLOCKS 256

//#define USE_AFFINE
