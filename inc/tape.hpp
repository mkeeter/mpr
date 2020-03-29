#pragma once
#include <cstdint>

#include "util.hpp"

// Forward declaration
namespace libfive {
class Tree;
}

namespace libfive {
namespace cuda {

struct Tape {
    Tape(const libfive::Tree& tree);

    // data is a pointer in GPU (unified) memory
    Ptr<uint64_t[]> data;
    int32_t length;
};

} // namespace cuda
} // namespace libfive
