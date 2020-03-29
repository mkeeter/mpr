#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "context.hpp"
#include "tape.hpp"
#include "gpu_opcode.hpp"
#include "clause.hpp"

// Not actually a benchmark, used to generate a figure for the paper
int main(int, char**)
{
    std::ifstream ifs;
    const auto filename = "../benchmark/files/prospero_long.frep";
    ifs.open(filename);
    libfive::Tree t(0.0);
    if (ifs.is_open()) {
        auto a = libfive::Archive::deserialize(ifs);
        t = a.shapes.front().tree;
    } else {
        std::cerr << "Could not open " << filename << "\n";
        exit(1);
    }

    const auto size = 1024;
    auto ctx = libfive::cuda::Context(size);
    auto tape = libfive::cuda::Tape(t);
    ctx.render2D(tape, Eigen::Matrix3f::Identity());

    // Save the image using libfive::Heightmap
    libfive::Heightmap out(size, size);
    unsigned i=0;
    for (int x=0; x < size; ++x) {
        for (int y=0; y < size; ++y) {
            out.depth(x, y) = ctx.stages[3].filled[i++];
        }
    }
    out.savePNG("hello_world.png");
    std::cout << "Initial clauses: " << tape.length << "\n";

    for (unsigned i=0; i < pow(size / 64, 2); ++i) {
        const auto tile = ctx.stages[0].tiles[i];
        if (tile.position == -1) {
            continue;
        }
        const uint32_t x = tile.position % (size / 64);
        const uint32_t y = tile.position / (size / 64);

        unsigned len = 0;
        for (auto j = tile.tape + 1; OP(&ctx.tape_data[j]); ++j) {
            auto d = ctx.tape_data[j];
            if (OP(&d) == libfive::cuda::GPU_OP_JUMP) {
                j += JUMP_TARGET(&d);
            } else {
                len++;
            }
        }
        if (len > 0xFFFFFF) {
            std::cout << "toooo big" << len << "\n";
            exit(1);
        }
        const uint32_t pix = 0xFF000000 | ((0xFFFFFF) & len);

        for (unsigned i=x*64; i < (x+1)*64; ++i) {
            for (unsigned j=y*64; j < (y+1)*64; ++j) {
                out.norm(j,i) = pix;
            }
        }
    }
    out.saveNormalPNG("tile_tape_lengths.png");
    out.norm = 0;

    for (unsigned i=0; i < ctx.stages[2].tile_array_size; ++i) {
        const auto tile = ctx.stages[2].tiles[i];
        if (tile.position == -1) {
            continue;
        }
        const uint32_t x = tile.position % (size / 8);
        const uint32_t y = tile.position / (size / 8);

        unsigned len = 0;
        for (auto j = tile.tape + 1; OP(&ctx.tape_data[j]); ++j) {
            auto d = ctx.tape_data[j];
            if (OP(&d) == libfive::cuda::GPU_OP_JUMP) {
                j += JUMP_TARGET(&d);
            } else {
                len++;
            }
        }
        if (len > 0xFFFFFF) {
            std::cout << "toooo big" << len << "\n";
            exit(1);
        }
        const uint32_t pix = 0xFF000000 | ((0xFFFFFF) & len);

        for (unsigned i=x*8; i < (x+1)*8; ++i) {
            for (unsigned j=y*8; j < (y+1)*8; ++j) {
                out.norm(j,i) = pix;
            }
        }
    }
    out.saveNormalPNG("subtile_tape_lengths.png");
}
