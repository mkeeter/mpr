#pragma once
#include <cstdint>

struct Clause {
    uint8_t opcode;
    uint8_t banks;
    uint16_t out;
    uint16_t lhs;
    uint16_t rhs;
};
