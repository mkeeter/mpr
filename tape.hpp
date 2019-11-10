#pragma once

#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>
#include "clause.hpp"

// The Tape is an on-device representation, so the pointers
// are returned from cudaMalloc.
struct Tape {
    ~Tape();
    Tape(Tape&& other);

    __host__ __device__ inline
    const Clause& operator[](uint16_t i) const { return tape[i]; }

    __host__ __device__ inline
    const float& constant(uint16_t i) const { return constants[i]; }

    static Tape build(libfive::Tree tree);
    const uint16_t num_clauses;
    const uint16_t num_constants;

    const uint16_t num_regs;
    const uint16_t num_csg_choices;
    struct Axes {
        uint16_t reg[3];
    };
    const Axes axes;

private:
    Tape(const char* data,
         uint16_t num_clauses, uint16_t num_constants,
         uint16_t num_regs, uint16_t num_csg_choices,
         Axes axes);

    Tape(const Tape& other)=delete;
    Tape& operator=(const Tape& other)=delete;

    /*  We allocate global data in bulk for the tape + constants */
    const char* data=nullptr;

    /*  These are pointers into data */
    const Clause* __restrict__ tape=nullptr;
    const float* __restrict__ constants=nullptr;
};
