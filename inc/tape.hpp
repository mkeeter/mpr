#pragma once
#include <cstdint>

#include "util.hpp"

namespace libfive {
class Tree;
namespace cuda {

struct Tape {
    Tape(const libfive::Tree& tree);

    // data is a pointer in GPU (unified) memory
    Ptr<uint64_t> data;
    int32_t length;
};

} // namespace cuda
} // namespace libfive
