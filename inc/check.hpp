#pragma once
#include <cstdio>
#include <cuda_runtime.h>

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
    return static_cast<T*>(ptr);
}
