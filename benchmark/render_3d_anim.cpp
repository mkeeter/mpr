/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

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

    auto tape = mpr::Tape(t);
    auto c = mpr::Context(resolution);
    mpr::Effects effects;

    libfive::Heightmap out(resolution, resolution);

    for (unsigned i=0; i < 360; ++i) {
        printf("%u\n", i);
        Eigen::Matrix3f T =
            Eigen::AngleAxisf(i * M_PI/180, Eigen::Vector3f(0, 0, 1)) *
            Eigen::AngleAxisf(60 * M_PI/180, Eigen::Vector3f(1, 0, 0)) *
            Eigen::Scaling(Eigen::Vector3f(1.2, 1.2, 2.0));
        Eigen::Matrix4f T_ = Eigen::Matrix4f::Identity();
        T_.topLeftCorner<3, 3>() = T;
        T_(3, 2) = 0.3f;  // Have some perspective
        T_(2, 3) = 0.25f; // Shift by Z a little
        c.render3D(tape, T_);
        effects.drawSSAO(c);
        effects.drawShaded(c);

        unsigned j=0;
        for (int x=0; x < resolution; ++x) {
            for (int y=0; y < resolution; ++y) {
                out.depth(x, y) = effects.image[j++];
            }
        }
        out.savePNG("frame" + std::to_string(i) + ".png");
    }

    return 0;
}


