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

#include "../libfive/libfive/test/util/shapes.hpp"

#include "context.hpp"
#include "tape.hpp"
#include "effects.hpp"

int main(int argc, char **argv)
{
    int resolution = 512;
    if (argc >= 2) {
        errno = 0;
        resolution = strtol(argv[1], NULL, 10);
        if (errno || resolution == 0) {
            fprintf(stderr, "Could not parse resolution '%s'\n",
                    argv[2]);
            exit(1);
        }
    }

    auto c = mpr::Context(resolution);
    mpr::Effects effects;

    for (unsigned i=0; i < 60; i++) {
        libfive::Tree t = box(Eigen::Vector3f(-1, -1, -2),
                              Eigen::Vector3f(1, 1, 2));

        // Cut out circle
        float f = i / 29.0f;
        if (i < 30) {
            f = -(cos(M_PI * f) - 1) / 2;
        } else {
            f = 1;
        }
        auto cutout = -circle(0.8 * f);
        t = max(t, cutout);

        if (i >= 30) {
            auto X = libfive::Tree::X();
            auto Y = libfive::Tree::Y();
            auto Z = libfive::Tree::Z();
            float f = (i - 30) / 29.0f;
            f = -(cos(M_PI * f) - 1) / 2;
            auto frac = Z * f;
            t = t.remap(
              (cos(frac) * X - sin(frac) * Y),
              (sin(frac) * X + cos(frac) * Y),
              Z);
        }

        auto tape = mpr::Tape(t);

        Eigen::Matrix3f T =
            Eigen::AngleAxisf(40, Eigen::Vector3f(0, 0, 1)) *
            Eigen::AngleAxisf(60 * M_PI/180, Eigen::Vector3f(1, 0, 0)) *
            Eigen::Scaling(Eigen::Vector3f(3, 3, 3));
        Eigen::Matrix4f T_ = Eigen::Matrix4f::Identity();
        T_.topLeftCorner<3, 3>() = T;
        T_(3, 2) = 0.3f;  // Have some perspective
        T_(2, 3) = 0.25f; // Shift by Z a little
        c.render3D(tape, T_);
        effects.drawSSAO(c);
        effects.drawShaded(c);

        unsigned j=0;
        libfive::Heightmap out(resolution, resolution);
        out.depth = 0;
        out.norm = 0;
        for (int x=0; x < c.image_size_px; ++x) {
            for (int y=0; y < c.image_size_px; ++y) {
                const auto p = c.stages[3].filled[j];
                out.depth(x, y) = p;
                if (p) {
                    out.norm(x, y) = c.normals[j];
                } else {
                    out.norm(x, y) = 0xFFFFFFFF;
                }
                ++j;
            }
        }
        auto s = std::to_string(i);
        while (s.size() < 3) {
            s = "0" + s;
        }
        std::cout << s << "\n";
        out.saveNormalPNG("frame" + s + ".png");
    }

    return 0;
}



