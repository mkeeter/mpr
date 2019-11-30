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
    uint32_t insert(uint32_t t) {
        const uint32_t out = atomicAdd(&count, 1);
        data[out] = t;
        return out;
    }
#endif

    __device__
    uint32_t operator[](uint32_t i) const { return data[i]; }

    uint32_t* __restrict__ data;
    uint32_t size;
    uint32_t count;
};
