#pragma once
#include <cstdint>

struct Clause {
    const uint8_t opcode;
    const uint8_t banks;
    const uint16_t out;
    const uint16_t lhs;
    const uint16_t rhs;
};
