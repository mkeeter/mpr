#include <cstdint>
#include "parameters.hpp"

struct Subtapes {
    /*  Each subtape is an array of LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE clauses.
     *
     *  subtape.data[i]'s valid clauses go from start[i] to the end of the
     *  array; if it's not full, the beginning is invalid (which is weird,
     *  but makes sense if you know how we're constructing subtapes). */
    uint32_t start[LIBFIVE_CUDA_NUM_SUBTAPES];
    uint32_t next[LIBFIVE_CUDA_NUM_SUBTAPES];
    Clause   data[LIBFIVE_CUDA_NUM_SUBTAPES]
                 [LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE];
};
