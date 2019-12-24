#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "renderable.hpp"

int main(int, char**)
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

    auto r = Renderable::build(t, 1024, 3);
    r->tape.print();

    r->run({Eigen::Matrix4f::Identity()});

    libfive::Heightmap out(r->image.size_px, r->image.size_px);
    out.norm = 0;
    for (unsigned x=0; x < r->image.size_px; ++x) {
        for (unsigned y=0; y < r->image.size_px; ++y) {
            if (r->heightAt(x, y)) {
                const auto o = static_cast<Renderable3D*>(r.get())->temp(x, y);
                out.norm(y, x) = (0xFF << 24) | (o << 16) | (o << 8) | o;
            }
        }
    }
    out.saveNormalPNG("out_gpu_ssao.png");
    /*
    // Save the image using libfive::Heightmap
    libfive::Heightmap out(r->image.size_px, r->image.size_px);
    for (unsigned x=0; x < r->image.size_px; ++x) {
        for (unsigned y=0; y < r->image.size_px; ++y) {
            out.depth(y, x) = r->heightAt(x, y);
        }
    }
    out.savePNG("out_gpu_depth.png");

    std::atomic_bool abort(false);
    libfive::Voxels vox({-1, -1, 0}, {1, 1, 0}, r->image.size_px / 2);
    auto start_cpu = std::chrono::steady_clock::now();
    for (unsigned i=0; i < 10; ++i) {
        auto h = libfive::Heightmap::render(t, vox, abort);
    }
    auto end_cpu = std::chrono::steady_clock::now();
    std::cout << "CPU rendering took " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() <<
        " ms\n";
    libfive::Heightmap::render(t, vox, abort)->savePNG("out_cpu.png");
    */

    return 0;
}

