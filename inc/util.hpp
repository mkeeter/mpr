/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <memory>

#define CUDA_CHECK(f) { gpuCheck((f), __FILE__, __LINE__); }
inline void gpuCheck(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define CUDA_MALLOC(T, c) cudaMallocManagedChecked<T>(c, __FILE__, __LINE__)
template <typename T>
inline T* cudaMallocManagedChecked(size_t count, const char *file, int line) {
    void* ptr;
    gpuCheck(cudaMallocManaged(&ptr, sizeof(T) * count), file, line);
    //printf("%p allocated %lu at [%s:%i]\n", ptr, sizeof(T) * count, file, line);
    return static_cast<T*>(ptr);
}

#define CUDA_FREE(c) cudaFreeChecked((void*)c, __FILE__, __LINE__)
inline void cudaFreeChecked(void* ptr, const char *file, int line) {
    //printf("%p freed [%s:%i]\n", ptr, file, line);
    gpuCheck(cudaFree(ptr), file, line);
}

namespace libfive {
namespace cuda {

struct Deleter {
    template <typename T>
    void operator()(T* ptr) { CUDA_FREE(ptr); }
};

template <typename T>
using Ptr = std::unique_ptr<T, Deleter>;

// Helper function to do constexpr integer powers
inline constexpr unsigned __host__ __device__ pow(unsigned p, unsigned n) {
    return n ? p * pow(p, n - 1) : 1;
}

}   // namespace libfive
}   // namespace cuda
