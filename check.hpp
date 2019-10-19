#pragma once
#include <cuda_runtime.h>

#define CHECK(f) { gpuCheck((f), __FILE__, __LINE__); }
inline void gpuCheck(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

template <typename T>
inline T* cudaMallocManagedChecked(size_t count) {
    void* ptr;
    CHECK(cudaMallocManaged(&ptr, sizeof(T) * count));
    return static_cast<T*>(ptr);
}
