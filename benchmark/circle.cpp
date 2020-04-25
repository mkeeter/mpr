/*
Reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
(SIGGRAPH 2020)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2019-2020  Matt Keeter
*/
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "context.hpp"
#include "tape.hpp"

// Not actually a benchmark, used to generate a figure for the paper
int main(int, char**)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto t = sqrt((X + 1)*(X + 1) + (Y + 1)*(Y + 1)) - 1.8;

    const auto size = 128;
    auto ctx = mpr::Context(128);
    auto tape = mpr::Tape(t);
    ctx.render2D(tape, Eigen::Matrix3f::Identity());

    libfive::Heightmap out(size, size);

    constexpr static auto COLOR_GREY = 0xFF636363;
    constexpr static auto COLOR_YELLOW = 0xFF21F9F0;
    constexpr static auto COLOR_ORANGE = 0xFF3E9AF8;
    constexpr static auto COLOR_BLUE = 0xFF880710;
    constexpr static auto COLOR_PINK = 0xFF6264E1;
    constexpr static auto COLOR_PURPLE = 0xFFA8006E;

    out.norm = COLOR_GREY; // Grey

    for (unsigned i=0; i < pow(size / 64, 2); ++i) {
        const auto tile = ctx.stages[0].tiles[i];
        if (tile.position != -1) {
            continue;
        }
        const uint32_t x = i % (size / 64);
        const uint32_t y = i / (size / 64);

        for (unsigned i=x*64; i < (x+1)*64; ++i) {
            for (unsigned j=y*64; j < (y+1)*64; ++j) {
                out.norm(j,i) = COLOR_YELLOW;
            }
        }
    }
    out.saveNormalPNG("circle1.png");

    for (unsigned i=0; i < ctx.stages[2].tile_array_size; ++i) {
        const auto tile = ctx.stages[2].tiles[i];
        if (tile.position != -1) {
            continue;
        }

        int q = -1;
        unsigned parent = -1;
        for (unsigned j=0; j < pow(size / 64, 2); ++j) {
            if (ctx.stages[0].tiles[j].position != -1) {
                q++;
            }
            if (q == (int)i / 64) {
                parent = ctx.stages[0].tiles[j].position;
                break;
            }
        }
        auto px = parent % (size / 64);
        auto py = parent / (size / 64);
        auto x = (px * 8) + ((i % 64) % 8);
        auto y = (py * 8) + ((i % 64) / 8);

        for (unsigned i=x*8; i < (x+1)*8; ++i) {
            for (unsigned j=y*8; j < (y+1)*8; ++j) {
                if (ctx.stages[3].filled[i + j * size]) {
                    out.norm(j,i) = COLOR_ORANGE;
                } else {
                    out.norm(j,i) = COLOR_BLUE;
                }
            }
        }
    }
    out.saveNormalPNG("circle2.png");

    for (unsigned i=0; i < size; ++i) {
        for (unsigned j=0; j < size; ++j) {
            if (out.norm(j, i) == COLOR_GREY) {
                if (ctx.stages[3].filled[i + j * size]) {
                    out.norm(j, i) = COLOR_PINK;
                } else {
                    out.norm(j, i) = COLOR_PURPLE;
                }
            }
        }
    }
    out.saveNormalPNG("circle3.png");
}
