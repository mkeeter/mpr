#include <cstdint>

#include "renderable_clause.hpp"
#include "renderable_parameters.hpp"

struct Subtapes {
    /*  Each subtape is an array of LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE clauses.
     *
     *  subtape.data[i]'s valid clauses go from start[i] to the end of the
     *  array; if it's not full, the beginning is invalid (which is weird,
     *  but makes sense if you know how we're constructing subtapes). */
    uint32_t start[LIBFIVE_CUDA_NUM_SUBTAPES];
    uint32_t next[LIBFIVE_CUDA_NUM_SUBTAPES];
    uint32_t prev[LIBFIVE_CUDA_NUM_SUBTAPES];
    Clause   data[LIBFIVE_CUDA_NUM_SUBTAPES]
                 [LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE];

    /* The index of the next available subtape chunk */
    uint32_t num_subtapes;

    void reset() {
        num_subtapes = 1;
    }

#ifdef __CUDACC__
    inline __device__ uint32_t claim() {
        const auto out = atomicAdd(&num_subtapes, 1);
        assert(out < LIBFIVE_CUDA_NUM_SUBTAPES);
        return out;
    }
#endif
};
