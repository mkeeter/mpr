/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "clause.hpp"
#include "tape.hpp"
#include "gpu_opcode.hpp"

// Not actually a benchmark, used to generate data for the paper
int main(int argc, char** argv)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto t = max(sqrt(X*X + Y*Y) - 1, 0.5 - sqrt(X*X + Y*Y));

    if (argc == 2) {
        std::ifstream ifs;
        ifs.open(argv[1]);
        if (ifs.is_open()) {
            auto a = libfive::Archive::deserialize(ifs);
            t = a.shapes.front().tree;
        } else {
            fprintf(stderr, "Could not open file %s\n", argv[1]);
            exit(1);
        }
    }
    auto r = libfive::cuda::Tape(t);

    for (int i=1; i < r.length - 1; ++i) {
        const auto c = r.data[i];
        std::cout << libfive::cuda::gpu_op_str(OP(&c)) << " & "
                  << (int)I_LHS(&c) << " & "
                  << (int)I_RHS(&c) << " & "
                  << IMM(&c) << (IMM(&c) == int(IMM(&c)) ? ".0f" : "f") << " & "
                  << (int)I_OUT(&c) << "\\\\\n";
    }
}
