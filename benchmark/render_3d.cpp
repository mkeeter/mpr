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
    auto r = Renderable::build(t, 1024, 3);
    r->tape.print();

    auto start_gpu = std::chrono::steady_clock::now();
    for (unsigned i=0; i < 10; ++i) {
        r->run({{0, 0, 0}, 1});
    }
    auto end_gpu = std::chrono::steady_clock::now();
    std::cout << "GPU rendering took " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() <<
        " ms\n";

    // Save the image using libfive::Heightmap
    libfive::Heightmap out(r->image.size_px, r->image.size_px);
    for (unsigned x=0; x < r->image.size_px; ++x) {
        for (unsigned y=0; y < r->image.size_px; ++y) {
            out.depth(y, x) = r->heightAt(x, y);
            out.norm(y, x) = r->normalAt(x, y);
        }
    }
    out.savePNG("out_gpu_depth.png");
    out.saveNormalPNG("out_gpu_norm.png");

    std::atomic_bool abort(false);
    const float z = LIBFIVE_CUDA_3D ? 1.0f : 0.0f;
    libfive::Voxels vox({-1, -1, -z}, {1, 1, z}, r->image.size_px / 2);
    auto start_cpu = std::chrono::steady_clock::now();
    for (unsigned i=0; i < 10; ++i) {
        auto h = libfive::Heightmap::render(t, vox, abort);
    }
    auto end_cpu = std::chrono::steady_clock::now();
    std::cout << "CPU rendering took " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() <<
        " ms\n";
    libfive::Heightmap::render(t, vox, abort)->savePNG("out_cpu.png");

    return 0;
}

