#pragma once

#include <cuda_runtime.h>
#include <libfive/tree/tree.hpp>
#include "clause.hpp"

// The Tape is an on-device representation, so the pointers
// are returned from cudaMalloc.
struct Tape {
    ~Tape();
    Tape(Tape&& other);

    __host__ __device__
    const Clause& operator[](uint32_t i) const { return data[i]; }

    __host__ __device__
    float constant(uint32_t i) const { return constants[i]; }

    static Tape build(libfive::Tree tree);
    const uint32_t tape_length;

    const uint16_t num_regs;
    const uint16_t num_csg_choices;

private:
    Tape(const Clause* data, uint32_t tape_length,
         uint16_t num_regs, uint16_t num_csg_choices,
         const float* constants);

    Tape(const Tape& other)=delete;
    Tape& operator=(const Tape& other)=delete;

    const Clause* __restrict__ data=nullptr;
    const float* __restrict__ constants=nullptr;
};
