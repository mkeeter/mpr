#pragma once

/*  Defines how many pixels are on the side of each tile */
#define LIBFIVE_CUDA_TILE_SIZE_PX 16

/*  Interval evaluation is done with 2D blocks
 *      LIBFIVE_CUDA_THREADS_PER_INTERVAL_BLOCK
 *  threads per side.
 *
 *  The number of blocks is calculated based on image size */
#define LIBFIVE_CUDA_THREADS_PER_INTERVAL_BLOCK 8

/*  Filling is done with a LIBFIVE_CUDA_NUM_FILL_BLOCKS grid (1D)  */
#define LIBFIVE_CUDA_NUM_FILL_BLOCKS 1024

/*  Evaluating ambiguous tiles is done with a
 *  LIBFIVE_CUDA_NUM_AMBIGUOUS_BLOCKS grid (1D)  */
#define LIBFIVE_CUDA_NUM_AMBIGUOUS_BLOCKS 1024

/*  This is the number of subtapes allocated.  Each subtape has room for some
 *  number of clauses, defined in the Subtape struct */
#define LIBFIVE_CUDA_NUM_SUBTAPES 65535

/*  This is the length of each subtape chunk */
#define LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE 250
