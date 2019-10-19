#pragma once

#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>
#include "clause.hpp"

// The Tape is an on-device representation, so the pointers
// are returned from cudaMalloc.
struct Tape {
    __host__ __device__
    const Clause& operator[](uint32_t i) const { return data[i]; }

    static Tape build(libfive::Tree tree);
    const Clause* const __restrict__ data;
    const uint32_t tape_length;

    const uint16_t num_regs;
    const uint16_t num_csg_choices;

    const float* const __restrict__ constants;
};
