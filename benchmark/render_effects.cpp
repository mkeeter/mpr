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
#include "effects.hpp"

int main(int argc, char** argv)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto Z = libfive::Tree::Z();
    auto t = sqrt(X*X + Y*Y + Z*Z) - 0.1;
    t = max(t, -max(-Z, max(X, Y)));

    // Rotate by a little bit about the X axis
    const auto angle = -M_PI/4;
    t = t.remap(X, cos(angle) * Y + sin(angle) * Z,
                  -sin(angle) * Y + cos(angle) * Z);

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
    }
    int resolution = 512;
    if (argc >= 3) {
        errno = 0;
        resolution = strtol(argv[2], NULL, 10);
        if (errno || resolution == 0) {
            fprintf(stderr, "Could not parse resolution '%s'\n",
                    argv[2]);
            exit(1);
        }
    }

    auto ctx = libfive::cuda::Context(resolution);
    auto tape = libfive::cuda::Tape(t);
    libfive::cuda::Effects effects;

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(3,2) = 0.3f;

    ctx.render3D(tape, T);
    effects.drawSSAO(ctx);

    libfive::Heightmap out(resolution, resolution);
    unsigned i=0;
    for (int x=0; x < resolution; ++x) {
        for (int y=0; y < resolution; ++y) {
            out.depth(x, y) = effects.image[i++];
        }
    }
    out.savePNG("out_gpu_ssao.png");

    effects.drawShaded(ctx);
    i=0;
    for (int x=0; x < resolution; ++x) {
        for (int y=0; y < resolution; ++y) {
            out.depth(x, y) = effects.image[i++];
        }
    }
    out.savePNG("out_gpu_shaded.png");

    return 0;
}
