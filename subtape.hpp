#pragma once
#include <cstdint>
#include "parameters.hpp"

struct Subtape {
    uint32_t next;
    uint32_t size;

    __host__ __device__
    uint32_t& operator[](uint32_t i) { return data[i]; }

    __host__ __device__
    const uint32_t& operator[](uint32_t i) const { return data[i]; }

protected:
    uint32_t data[LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE];
};
