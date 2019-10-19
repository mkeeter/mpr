#pragma once
#include <cstdint>

struct Subtape {
    uint32_t next;
    uint32_t size;
    uint32_t subtape[256 - 2];
};
