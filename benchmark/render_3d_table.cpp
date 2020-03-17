#include <cstdio>
#include <chrono>
#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "renderable.hpp"
#include "v3.hpp"

#include "tape.hpp"
#include "context.hpp"

void get_stats(std::function<void()> f) {
    // Warm up
    for (unsigned i=0; i < 20; ++i) {
        f();
    }
    std::vector<double> times_ms;
    const auto count = 100;
    for (unsigned i=0; i < count; ++i) {
        using std::chrono::steady_clock;
        using std::chrono::duration_cast;
        using std::chrono::nanoseconds;
        auto start_gpu = steady_clock::now();
        f();
        auto end_gpu = steady_clock::now();
        times_ms.push_back(
                duration_cast<nanoseconds>(end_gpu - start_gpu).count() / 1e6);
    }
    double mean = 0;
    for (auto& b : times_ms) {
        mean += b;
    }
    mean /= times_ms.size();
    double stdev = 0;
    for (auto& b : times_ms) {
        stdev += std::pow(b - mean, 2);
    }
    stdev = sqrt(stdev / (times_ms.size() - 1));
    std::cout << mean << " " << stdev << "\n";
}

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

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(3,2) = 0.3f;

    const std::vector<int> sizes = {256, 512};//, 1024, 1536, 2048};
    std::cout << "Rendering with v3 architecture:" << std::endl;
    for (auto size: sizes) {
        auto r = build_v3_blob(t, size);

        std::cout << size << " ";
        get_stats([&](){ render_v3_blob(r, Eigen::Matrix4f::Identity()); });

        libfive::Heightmap out(size, size);
        uint32_t i = 0;
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out.depth(y, x) = r.stages[3].filled[i++];
            }
        }
        out.savePNG("out_gpu_depth_v3_" + std::to_string(size) + ".png");
        free_v3_blob(r);
    }
    std::cout << "Rendering with context architecture:" << std::endl;
    for (auto size: sizes) {
        auto tape = libfive::cuda::Tape(t);
        auto c = libfive::cuda::Context(size);

        std::cout << size << " ";
        get_stats([&](){ c.render3D(tape, Eigen::Matrix4f::Identity()); });

        libfive::Heightmap out(size, size);
        uint32_t i = 0;
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out.depth(y, x) = c.stages[3].filled.get()[i++];
            }
        }
        out.savePNG("out_gpu_depth_ctx_" + std::to_string(size) + ".png");
    }
    std::cout << "Rendering with original architecture:" << std::endl;
    for (auto size: sizes) {
        auto r = Renderable::build(t, size, 3);
        std::cout << size << " ";
        get_stats([&](){r->run({Eigen::Matrix4f::Identity()}, Renderable::MODE_SHADED);});

        libfive::Heightmap out(size, size);
        for (int x=0; x < size; ++x) {
            for (int y=0; y < size; ++y) {
                out.depth(y, x) = r->heightAt(y, x);
            }
        }
        out.savePNG("out_gpu_depth_orig_" + std::to_string(size) + ".png");
    }
    return 0;
}
