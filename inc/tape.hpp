/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
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
