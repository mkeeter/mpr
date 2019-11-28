#include "check.hpp"

struct Queue {
    Queue()
        : data(nullptr), size(0), count(0)
    {
        // Nothing to do here
    }

    void resizeToFit(uint32_t num)
    {
        if (num > size) {
            CUDA_CHECK(cudaFree(data));
            data = CUDA_MALLOC(uint32_t, num);
            size = num;
        }
        count = 0;
    }

#ifdef __CUDACC__
    __device__
    void insert(uint32_t t) {
        data[atomicAdd(&count, 1)] = t;
    }
#endif

    __device__
    uint32_t operator[](uint32_t i) const { return data[i]; }

    uint32_t* __restrict__ data;
    uint32_t size;
    uint32_t count;
};
