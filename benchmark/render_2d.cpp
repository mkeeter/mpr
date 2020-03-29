/*
libfive-cuda: a GPU-accelerated renderer for libfive

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "context.hpp"
#include "tape.hpp"

int main(int argc, char **argv)
{
    libfive::Tree t = libfive::Tree::X();
    if (argc >= 2) {
        std::ifstream ifs;
        ifs.open(argv[1]);
        if (ifs.is_open()) {
            auto a = libfive::Archive::deserialize(ifs);
            t = a.shapes.front().tree;
        } else {
            fprintf(stderr, "Could not open file %s\n", argv[1]);
            exit(1);
        }
    } else {
        auto X = libfive::Tree::X();
        auto Y = libfive::Tree::Y();
        auto Z = libfive::Tree::Z();
        t = min(sqrt((X + 0.5)*(X + 0.5) + Y*Y + Z*Z) - 0.25,
                sqrt((X - 0.5)*(X - 0.5) + Y*Y + Z*Z) - 0.25);
    }
    int resolution = 2048;
    if (argc >= 3) {
        errno = 0;
        resolution = strtol(argv[2], NULL, 10);
        if (errno || resolution == 0) {
            fprintf(stderr, "Could not parse resolution '%s'\n",
                    argv[2]);
            exit(1);
        }
    }

    auto tape = libfive::cuda::Tape(t);
    auto c = libfive::cuda::Context(resolution);

    c.render2D(tape, Eigen::Matrix3f::Identity(), 0.0f);

    // Save the image using libfive::Heightmap
    libfive::Heightmap out(c.image_size_px, c.image_size_px);
    unsigned i=0;
    for (int x=0; x < c.image_size_px; ++x) {
        for (int y=0; y < c.image_size_px; ++y) {
            out.depth(x, y) = c.stages[3].filled[i++];
        }
    }
    out.savePNG("out_gpu_depth.png");

    std::atomic_bool abort(false);
    libfive::Voxels vox({-1, -1, 0}, {1, 1, 0}, c.image_size_px / 2);
    auto h = libfive::Heightmap::render(t, vox, abort);
    libfive::Heightmap::render(t, vox, abort)->savePNG("out_cpu.png");

    return 0;
}
