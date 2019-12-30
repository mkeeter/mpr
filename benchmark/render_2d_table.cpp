#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "renderable.hpp"

int main(int argc, char **argv)
{
    libfive::Tree t = libfive::Tree::X();
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
    } else {
        auto X = libfive::Tree::X();
        auto Y = libfive::Tree::Y();
        auto Z = libfive::Tree::Z();
        t = min(sqrt((X + 0.5)*(X + 0.5) + Y*Y + Z*Z) - 0.25,
                sqrt((X - 0.5)*(X - 0.5) + Y*Y + Z*Z) - 0.25);
    }
    for (auto size: {256, 512, 1024, 2048, 3074, 4096}) {
        auto r = Renderable::build(t, size, 2);
        // Warm-up
        for (unsigned i=0; i < 20; ++i) {
            r->run({Eigen::Matrix4f::Identity()}, Renderable::MODE_HEIGHTMAP);
        }
        auto start_gpu = std::chrono::steady_clock::now();
        const auto count = 100;
        for (unsigned i=0; i < count; ++i) {
            r->run({Eigen::Matrix4f::Identity()}, Renderable::MODE_HEIGHTMAP);
        }
        auto end_gpu = std::chrono::steady_clock::now();
        std::cout << size << " " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() / count <<
            " ms\n";
    }
    return 0;
}

